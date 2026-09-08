"""
Modal entrypoints, one per stage. Verified against modal 1.5.0.

  modal run modal_app.py::stage1_data
  modal run modal_app.py::stage2_train --variant aux --epochs 12
  modal run modal_app.py::stage3_gates --ckpt /runs/aux/ep12.pt
  modal run modal_app.py::stage3_sweep          # 3 full runs, one per SIGReg lambda
  modal run modal_app.py::stage4_plan --ckpt ...

Compute split: data-gen is emulator-bound so it fans out across CPU containers;
training/gates/SAE/planning are GPU. Everything lands on Volumes so the stages
compose. aq/aquin run alongside as the record-and-inspect layer — they provide
no compute of their own (`aq job run --gpu` is a LOCAL subprocess runner).
"""
import modal
from pathlib import Path

# Two images. Stage 1 is emulator-bound and needs NO torch — pulling the full
# CUDA stack for it made the build ~4 GB and timed out on save. Splitting cuts
# the data-gen image to the emulator plus opencv, so Stage 1 builds in seconds.
EMULATOR_PINS = [
    # Each pin found by a real failure, not caution:
    #   numpy<2      nes-py's _rom.py does uint8 * 2**10, which overflows under
    #                NumPy 2's NEP-50 scalar promotion (OverflowError)
    #   gym==0.25.2  0.26 switched step() to a 5-tuple; nes-py returns 4
    #   nes-py 8.2.1 the version that builds against the above
    "numpy<2", "gym==0.25.2", "nes-py==8.2.1", "gym-super-mario-bros==7.4.0",
]

cpu_image = (
    modal.Image.debian_slim()
    .pip_install(*EMULATOR_PINS, "opencv-python-headless", "tqdm")
    .add_local_dir("aqmario", "/root/aqmario")
    .add_local_dir("scripts", "/root/scripts")
)

# PPO needs the emulator AND torch AND gymnasium (the API adapter in
# aqmario/ppo.py bridges nes-py's old 4-tuple to SB3's 5-tuple), but not timm
# or aquin. opencv is the preprocessing.
ppo_image = (
    modal.Image.debian_slim()
    .pip_install(*EMULATOR_PINS, "opencv-python-headless", "tqdm",
                 "torch", "gymnasium", "stable-baselines3", "tensorboard")
    .add_local_dir("aqmario", "/root/aqmario")
    .add_local_dir("scripts", "/root/scripts")
)

gpu_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install(*EMULATOR_PINS, "opencv-python-headless", "tqdm",
                 "torch", "timm", "stable-baselines3", "aquin")
    .add_local_dir("aqmario", "/root/aqmario")
    .add_local_dir("scripts", "/root/scripts")
)

app = modal.App("aq-mario", image=gpu_image)
data_vol = modal.Volume.from_name("aqmario-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("aqmario-runs", create_if_missing=True)
VOLS = {"/data": data_vol, "/runs": runs_vol}

# aquin watch ingest + sae train both call require_active_session, so Stage 2/3
# containers need the CLI token. Create with:
#   modal secret create aquin-token AQUIN_TOKEN=aq-...
# Stage 1 does not need it, and a missing Stage 2 dependency must not block
# Stage 1 — so resolve it lazily and degrade to no secret.
def _aquin_secret():
    try:
        s = modal.Secret.from_name("aquin-token")
        s.hydrate()
        return [s]
    except Exception:
        print("[modal_app] aquin-token secret absent; aquin watch/SAE sync disabled")
        return []


AQUIN = _aquin_secret()


# --- Stage 1: ground truth -- CPU, fan out ----------------------------------
@app.function(image=cpu_image, volumes=VOLS, cpu=2.0, timeout=60 * 60, max_containers=64)
def collect_shard(policy: str, seed: int, episodes: int):
    import subprocess, sys
    subprocess.run([sys.executable, "/root/scripts/collect_data.py",
                    "--policy", policy, "--episodes", str(episodes),
                    "--shard-start", str(seed * 1000),
                    "--out-dir", "/data"], check=True)
    data_vol.commit()


@app.local_entrypoint()
def stage1_data(episodes_per_shard: int = 250, shards: int = 16, policy: str = "random"):
    """
    Fan out data collection. Default is the random track only: the ppo track
    needs a trained SB3 checkpoint that scripts/train_ppo.py has not produced
    yet, and random alone does not reach the back half of 1-1 (see STAGES.md).
    """
    jobs = [(policy, s, episodes_per_shard) for s in range(shards)]
    list(collect_shard.starmap(jobs))
    print(f"stage 1: {shards * episodes_per_shard} {policy} episodes requested")


# The tracker probes local state and knows nothing about Modal Volumes, so
# without this it reported "3 shards / 48 episodes" while 4000 episodes sat in
# aqmario-data. This writes a manifest it can read.
@app.function(image=cpu_image, volumes=VOLS, cpu=2.0, timeout=30 * 60)
def _data_stats():
    import numpy as np
    from pathlib import Path as P
    out = {}
    total_bytes = 0
    for sub in sorted(P("/data").iterdir()):
        if not sub.is_dir():
            continue
        shards = sorted(sub.glob("shard_*.npz"))
        eps = obs = 0
        for s in shards:
            total_bytes += s.stat().st_size
            with np.load(s) as d:                 # ep_offsets only: no frame inflation
                off = d["ep_offsets"]
            eps += len(off) - 1
            obs += int(off[-1])
        out[sub.name] = {"shards": len(shards), "episodes": int(eps), "observations": int(obs)}
    return {"by_policy": out, "gb": total_bytes / 1e9}


@app.local_entrypoint()
def data_stats():
    import json, time
    from pathlib import Path
    res = _data_stats.remote()
    res["taken_at"] = time.time()
    res["stale_hours"] = 0.0
    d = Path("~/.aqmario/data").expanduser()
    d.mkdir(parents=True, exist_ok=True)
    (d / "modal_manifest.json").write_text(json.dumps(res, indent=2))
    for k, v in sorted(res["by_policy"].items()):
        print(f"  {k:<8} {v['shards']:>4} shards  {v['episodes']:>5} episodes  "
              f"{v['observations']:>9} observations")
    print(f"  total {res['gb']:.1f} GB on the volume")


# The PPO explorer. A10G, not H100: PPO here is bounded by 8 emulator processes
# on the CPU side, so the GPU is mostly idle no matter which one you rent.
@app.function(image=ppo_image, volumes=VOLS, gpu="A10G", cpu=8.0, memory=16384,
              timeout=6 * 60 * 60)
def train_ppo(steps: int = 2_000_000, n_envs: int = 8, target_x: int = 2400):
    import json, subprocess, sys
    from pathlib import Path as P
    subprocess.run([sys.executable, "/root/scripts/train_ppo.py",
                    "--steps", str(steps), "--n-envs", str(n_envs),
                    "--target-x", str(target_x),
                    "--out", "/runs/ppo/ppo_mario.zip"], check=True)
    runs_vol.commit()
    cov = P("/runs/ppo/ppo_coverage.json")
    return json.loads(cov.read_text()) if cov.is_file() else {}


@app.local_entrypoint()
def stage1_ppo(steps: int = 2_000_000, n_envs: int = 8, target_x: int = 2400):
    import json
    from pathlib import Path
    res = train_ppo.remote(steps=steps, n_envs=n_envs, target_x=target_x)
    # Mirror locally: the tracker probes real state and cannot see Modal Volumes,
    # so without this it reports "PPO not trained" after a successful run.
    out = Path("~/.aqmario/runs/ppo").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "ppo_coverage.json").write_text(json.dumps(res, indent=2))
    print(res)


# collect_shard for the ppo track needs the SB3 checkpoint, so it runs on
# ppo_image (which has torch) rather than the lean cpu_image.
@app.function(image=ppo_image, volumes=VOLS, cpu=2.0, timeout=60 * 60, max_containers=64)
def collect_shard_ppo(seed: int, episodes: int, ckpt: str = "/runs/ppo/ppo_mario.zip"):
    import subprocess, sys
    subprocess.run([sys.executable, "/root/scripts/collect_data.py",
                    "--policy", "ppo", "--episodes", str(episodes),
                    "--shard-start", str(seed * 1000),
                    "--ppo-ckpt", ckpt, "--out-dir", "/data"], check=True)
    data_vol.commit()


@app.local_entrypoint()
def stage1_data_ppo(episodes_per_shard: int = 250, shards: int = 16):
    list(collect_shard_ppo.starmap([(s, episodes_per_shard) for s in range(shards)]))
    print(f"stage 1: {shards * episodes_per_shard} ppo episodes requested")


# --- Stage 2: world model -- one GPU ----------------------------------------
@app.function(volumes=VOLS, gpu="H100", cpu=16.0, memory=65536,
              timeout=8 * 60 * 60, secrets=AQUIN)
def train(variant: str = "aux", epochs: int = 3, sigreg_lambda: float = 10.0,
          batch_size: int = 96, stop_grad: bool = True, aux_scale: float = 20.0):
    """
    cpu=16 is not optional: at 224px the loader must decompress ~200 GB/epoch,
    and one core decompresses at ~300 MB/s. Fewer cores idles the H100.
    """
    import sys
    sys.path.insert(0, "/root")
    from aqmario.config import CFG
    from aqmario.train import train_jepa

    # Defaults are the measured winners, not the plan's originals. stop_grad
    # True and sigreg_lambda 10 (not 0.1) is what the dry-run grid says keeps
    # eff_dim rising: at lam 0.1/no-stop-grad the latent goes to rank 2 while
    # gain_5step reads 0.97, which looks like beating LeMario and is not.
    CFG.loss.variant = variant
    CFG.loss.sigreg_lambda = sigreg_lambda
    CFG.loss.aux_scale = aux_scale
    tag = f"{variant}_lam{sigreg_lambda}_aux{aux_scale:g}_{'sg' if stop_grad else 'nosg'}"
    CFG.run_dir = Path(f"/runs/{tag}")
    s = train_jepa("/data", out_weights=f"/runs/{tag}/jepa.pt", epochs=epochs,
                   batch_size=batch_size, num_workers=12,
                   run_dir=f"/runs/{tag}", cfg=CFG,
                   stop_grad_target=stop_grad,
                   checkpoint_every_frac=0.10,   # 10 look-ins, not 3
                   gate_grace_frac=0.35)         # probes need ~a third of the run
    s["ckpt"] = f"/runs/{tag}/jepa.pt"
    # The 10% health table is the thing to look at before trusting anything else.
    print(f"\n{'%':>5}{'y_probe':>10}{'x_probe':>10}{'pred':>9}{'gain5':>8}"
          f"{'eff_dim':>9}{'sigreg':>8}   health")
    for g in s.get("gate_by_epoch", []):
        print(f"{int(g['frac']*100):>4}%{g.get('y_probe_r2', float('nan')):>10.4f}"
              f"{g.get('x_probe_r2', float('nan')):>10.4f}{g.get('pred_loss', float('nan')):>9.4f}"
              f"{g.get('gain_5step', float('nan')):>8.3f}{g.get('eff_dim', float('nan')):>9.1f}"
              f"{g.get('sigreg_ratio', float('nan')):>7.1f}x   "
              f"{'ok' if g.get('healthy') else 'WARN'}")
    s.pop("history", None)
    runs_vol.commit()
    print(s)
    return s


@app.local_entrypoint()
def stage2_train(variant: str = "aux", epochs: int = 3, sigreg_lambda: float = 10.0,
                 batch_size: int = 96, stop_grad: bool = True, aux_scale: float = 20.0):
    s = train.remote(variant=variant, epochs=epochs, sigreg_lambda=sigreg_lambda,
                     batch_size=batch_size, stop_grad=stop_grad, aux_scale=aux_scale)
    print(s)
    print(f"\nnow gate it:\n  modal run modal_app.py::stage3_gates --ckpt {s.get('ckpt')}")


# The dry run is the Stage 2 exit check for "does it learn at all", and it is
# deliberately on an A10G rather than the H100: it is 200 steps on one shard, so
# it is bounded by the data loader, not the matmuls, and the H100 would idle at
# 10x the price.
@app.function(volumes=VOLS, gpu="A10G", cpu=8.0, memory=32768, timeout=60 * 60)
def dryrun(steps: int = 300, batch_size: int = 32, variant: str = "aux",
           sigreg_lambda: float = 0.1, stop_grad: bool = False,
           aux_scale: float = 1.0):
    import json, sys
    sys.path.insert(0, "/root")
    from pathlib import Path as P
    from aqmario.config import CFG
    from aqmario.train import dry_run

    CFG.loss.variant = variant
    CFG.loss.sigreg_lambda = sigreg_lambda
    CFG.loss.aux_scale = aux_scale
    # PER-CELL run dir. Every cell of a sweep sharing /runs/dryrun means they
    # overwrite each other's dryrun.pt and metrics.jsonl, and whichever container
    # happens to finish last is the checkpoint you end up running gates on —
    # silently, and not necessarily the cell you meant.
    tag = (f"{variant}_b{batch_size}_lam{sigreg_lambda}_aux{aux_scale:g}"
           f"_{'sg' if stop_grad else 'nosg'}")
    rd = P(f"/runs/dryrun/{tag}")
    CFG.run_dir = rd
    res = dry_run("/data", steps=steps, batch_size=batch_size,
                  num_workers=6, cfg=CFG, run_dir=str(rd),
                  stop_grad_target=stop_grad)
    res["aux_scale"] = aux_scale
    res["tag"] = tag
    res["ckpt"] = str(rd / "dryrun.pt")
    runs_vol.commit()
    return res


@app.local_entrypoint()
def stage2_dryrun_sweep(steps: int = 500, batch_size: int = 32,
                        lambdas: str = "1.0", batches: str = "", stop_grads: str = "0",
                        aux_scales: str = "1", variants: str = "aux"):
    """
    The dry run across a grid, in parallel. This is the cheap version of the
    Stage 3 lambda sweep and it answers the only question worth answering before
    spending H100 hours: is there ANY setting at which this does not collapse?

    Three axes, because the first sweep showed lambda alone cannot fix it:
      lambdas    sigreg strength
      batches    SIGReg's collapse sensitivity is mostly a function of BATCH
                 SIZE — measured, a rank-2 latent reads 2.8x null at n=32 and
                 41.8x at n=512, against a healthy 0.4x / 1.0x. At batch 32 the
                 statistic barely separates collapse from health, so it cannot
                 push back on it either.
      stop_grads with a shared encoder and no stop-gradient, MSE(zhat, z_target)
                 is minimised by making z_target CONSTANT — the target path is
                 trainable. LeJEPA claims SIGReg alone suffices; this is the
                 ablation that says whether that holds at our batch size.
    """
    import json
    from pathlib import Path

    lams = [float(x) for x in lambdas.split(",") if x.strip()]
    bs = [int(x) for x in batches.split(",") if x.strip()] or [batch_size]
    sgs = [bool(int(x)) for x in stop_grads.split(",") if x.strip()]
    axs = [float(x) for x in aux_scales.split(",") if x.strip()]
    vrs = [v.strip() for v in variants.split(",") if v.strip()]
    # variant "pure" drops the aux heads from the LOSS entirely (they stay built,
    # for the gates). This is the control that matters for the LeMario
    # comparison: the aux variant trains an explicit MSE(y_head(z), y) term, so
    # measuring "y is decodable from z" on it is partly measuring the thing we
    # asked for. LeMario had no such term.
    grid = [(steps, b, v, l, sg, a)
            for v in vrs for l in lams for b in bs for sg in sgs for a in axs]
    print(f"{len(grid)} cells: variants={vrs} lambdas={lams} batches={bs} "
          f"stop_grads={sgs} aux_scales={axs}")
    res = list(dryrun.starmap(grid))

    out = Path("~/.aqmario/runs").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "dryrun_sweep.json").write_text(json.dumps(res, indent=2))

    print(f"\n{'var':>6}{'lam':>6}{'aux':>6}{'batch':>7}{'sg':>6}{'pred':>18}{'gain_5':>9}"
          f"{'eff_dim':>16}{'sigreg':>10}   verdict")
    for r in sorted(res, key=lambda r: -r["eff_dim_frac_of_null"]):
        ed = f"{r['eff_dim_last']:.1f}/{r['eff_dim_null']:.0f}"
        ed = f"{r['eff_dim_first']:.1f}->{r['eff_dim_last']:.1f}"
        print(f"{r.get('variant','?')[:4]:>6}{r['sigreg_lambda']:>6.2f}{r.get('aux_scale', 1):>6.0f}{r['batch_size']:>7}"
              f"{str(r.get('stop_grad_target'))[0]:>6}"
              f"{r['pred_loss_first']:>8.3f}->{r['pred_loss_last']:<9.3f}"
              f"{r.get('gain_5step_last', float('nan')):>9.3f}"
              f"{ed:>16}{r['sigreg_ratio_last']:>9.1f}x   "
              f"{'PASS' if r['passed'] else 'FAIL ' + ','.join(k[:4] for k in ('not_collapsed','regularised','pred_loss_moved') if not r[k])}")
    # the tracker reads dryrun.json; give it the best PASSING run, else the best
    # eff_dim, so the board never claims a pass that did not happen.
    best = max(res, key=lambda r: (r["passed"], r["eff_dim_frac_of_null"]))
    (out / "dryrun.json").write_text(json.dumps(best, indent=2))
    if best.get("params"):
        (out / "param_count.json").write_text(json.dumps({"total": best["params"]}, indent=2))
    print(f"\nwrote dryrun.json from {best.get('tag')} "
          f"({'PASS' if best['passed'] else 'FAIL'})")
    if best.get("ckpt"):
        print(f"run the gates on it:\n"
              f"  modal run modal_app.py::stage3_gates --ckpt {best['ckpt']}")


@app.local_entrypoint()
def stage2_dryrun(steps: int = 400, batch_size: int = 32):
    import json
    from pathlib import Path
    res = dryrun.remote(steps=steps, batch_size=batch_size)
    # Mirror the result into the LOCAL run dir so `python -m aqmario.tracker`
    # sees the Modal run. The tracker probes real state and does not know about
    # Modal Volumes, so without this the board reports a dry run that happened.
    out = Path("~/.aqmario/runs").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "dryrun.json").write_text(json.dumps(res, indent=2))
    if res.get("params"):
        (out / "param_count.json").write_text(
            json.dumps({"total": res["params"]}, indent=2))
    print(json.dumps(res, indent=2))


# --- Stage 3: gates + inspect -- cheaper GPU --------------------------------
@app.function(volumes=VOLS, gpu="A10G", cpu=4.0, timeout=2 * 60 * 60, secrets=AQUIN)
def gates(ckpt: str):
    import json, subprocess, sys
    from pathlib import Path as P
    tag = P(ckpt).parent.name
    out = P(f"/runs/{tag}/gates_{tag}.json")
    before = out.stat().st_mtime if out.is_file() else 0
    r = subprocess.run([sys.executable, "/root/scripts/run_gates.py",
                        "--ckpt", ckpt, "--data-dir", "/data",
                        "--out", str(out), "--tag", tag])
    runs_vol.commit()
    # exit 1 means a gate FAILED, which is a RESULT and must be returned.
    # Any other exit code is the script crashing, and returning the stale JSON
    # from a previous run would report last week's numbers as this run's — which
    # is exactly what happened once and is far worse than an error.
    if r.returncode not in (0, 1):
        raise RuntimeError(f"run_gates.py crashed with exit {r.returncode}")
    if not out.is_file() or out.stat().st_mtime == before:
        raise RuntimeError(f"{out} was not rewritten — refusing to return stale gate results")
    return json.loads(out.read_text())


@app.function(volumes=VOLS, gpu="A10G", cpu=4.0, timeout=2 * 60 * 60, secrets=AQUIN)
def sae(ckpt: str, tag: str = "base", layer: int = 0, max_steps: int = 4000):
    """Dump latents (+ RAM labels) -> chunk_*.pt, then train the 192-dim SAE."""
    import json, sys
    sys.path.insert(0, "/root")
    from aqmario.aq_sae import dump_latents_with_labels, feature_report, train_latent_sae
    d = dump_latents_with_labels(ckpt, out_dir=f"/runs/lat/{tag}", data_dir="/data")
    out = f"/runs/sae/{tag}.pt"
    train_latent_sae(d, tag=tag, layer=layer, output=out, max_steps=max_steps)
    rep = feature_report(out, d)
    rep.pop("corr", None); rep.pop("live_idx", None)      # large, stays in the SAE dir
    runs_vol.commit()
    return {"sae": out, "acts": str(d), **rep}


@app.function(volumes=VOLS, gpu="A10G", cpu=4.0, timeout=2 * 60 * 60, secrets=AQUIN)
def sae_lambda_diff(tags: str):
    """tags: comma-separated 'lambda=tag' pairs, e.g. '0.1=aux_lam0.1,10=aux_lam10'."""
    import json, sys
    sys.path.insert(0, "/root")
    from aqmario.aq_sae import lambda_diff
    pairs = dict(p.split("=", 1) for p in tags.split(","))
    saes = {k: f"/runs/sae/{v}.pt" for k, v in pairs.items()}
    acts = {k: f"/runs/lat/{v}" for k, v in pairs.items()}
    out = lambda_diff(saes, acts, output="/runs/sae/lambda_diff.json")
    runs_vol.commit()
    return json.loads(open(out).read())


@app.local_entrypoint()
def stage3_sae_diff(tags: str):
    import json
    from pathlib import Path
    res = sae_lambda_diff.remote(tags)
    o = Path("~/.aqmario/runs/sae").expanduser(); o.mkdir(parents=True, exist_ok=True)
    (o / "lambda_diff.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res.get("per_lambda", {}), indent=2))
    for p in res.get("pairs", []):
        print(f"lam {p['lambda_a']} vs {p['lambda_b']}: "
              f"{p['orphaned_a']}/{p['live_a']} features orphaned, "
              f"y-corr matched {p['mean_y_corr_matched']} vs orphaned {p['mean_y_corr_orphaned']}")


@app.local_entrypoint()
def stage3_gates(ckpt: str):
    import json
    from pathlib import Path
    res = gates.remote(ckpt)
    out = Path("~/.aqmario/runs").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "gates_modal.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


@app.local_entrypoint()
def stage3_sweep(epochs: int = 12):
    """
    The SIGReg lambda-sweep. NOTE: each lambda is a FULL training run
    (~3.2 h H100, ~$17), not a cheap SAE refit. Modal runs them in parallel,
    so this costs money but no extra wall clock.
    """
    from aqmario.config import CFG
    lams = CFG.loss.sigreg_sweep
    print(f"launching {len(lams)} full training runs: lambda={lams}")
    ckpts = list(train.starmap([("aux", epochs, l) for l in lams]))
    for c, l in zip(ckpts, lams):
        sae.spawn(c, tag=f"lam{l}", layer=int(l * 100))
    print(ckpts)


# --- Stage 4: control -------------------------------------------------------
@app.function(volumes=VOLS, gpu="A10G", cpu=4.0, timeout=4 * 60 * 60)
def plan(ckpt: str, sae_path: str = "", episodes: int = 3):
    """Emulator + GPU in one container: CEM rolls out in latent space, the
    chosen macro executes on the real env, repeat."""
    import json, subprocess, sys
    from pathlib import Path as P
    tag = P(ckpt).parent.name
    out = P(f"/runs/{tag}")
    cmd = [sys.executable, "/root/scripts/run_plan.py", "--ckpt", ckpt,
           "--data-dir", "/data", "--out-dir", str(out),
           "--episodes", str(episodes)]
    if sae_path:
        cmd += ["--sae", sae_path]
    subprocess.run(cmd, check=True)
    runs_vol.commit()
    res = {}
    for name in ("plan_best.json", "plan_sanity.json", "steer_demo.json"):
        f = out / name
        if f.is_file():
            res[name.replace(".json", "")] = json.loads(f.read_text())
    return res


@app.local_entrypoint()
def stage4_plan(ckpt: str, sae_path: str = "", episodes: int = 3):
    import json
    from pathlib import Path
    res = plan.remote(ckpt, sae_path, episodes)
    # Mirror to the local run dir so the tracker can see Stage 4 happened.
    out = Path("~/.aqmario/runs").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for k, v in res.items():
        (out / f"{k}.json").write_text(json.dumps(v, indent=2))
    print(json.dumps(res, indent=2))


@app.function(image=cpu_image, volumes=VOLS, cpu=2.0, timeout=15 * 60)
def smoke():
    """Cheap preflight: does the pinned emulator stack actually work in-image?"""
    import sys
    sys.path.insert(0, "/root")
    import numpy as np, gym, gym_super_mario_bros
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    from aqmario import ram as R
    env = JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-1-v0"), COMPLEX_MOVEMENT)
    env.reset()
    for _ in range(30):
        out = env.step(3)
    m = env.unwrapped.ram
    x = R._pair(m, 0x006D, 0x0086)
    env.close()
    # Everything returned across the Modal boundary must be a PLAIN python type.
    # `info["x_pos"]` is a numpy scalar, and unpickling it locally requires numpy
    # in the *modal CLI's* env (a uv tool install, not this repo's venv) — which
    # fails with DeserializationError and hides an otherwise successful run.
    return dict(numpy=str(np.__version__), gym=str(gym.__version__),
                step_tuple=int(len(out)), world_x=int(x),
                info_x=int(out[3].get("x_pos", -1)))


@app.local_entrypoint()
def preflight():
    print(smoke.remote())
