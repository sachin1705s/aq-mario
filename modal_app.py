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

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install(
        "gym-super-mario-bros==7.4.0", "nes-py", "opencv-python-headless",
        "numpy", "tqdm", "torch", "timm", "stable-baselines3", "aquin",
    )
    .add_local_dir("aqmario", "/root/aqmario")
    .add_local_dir("scripts", "/root/scripts")
)

app = modal.App("aq-mario", image=image)
data_vol = modal.Volume.from_name("aqmario-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("aqmario-runs", create_if_missing=True)
VOLS = {"/data": data_vol, "/runs": runs_vol}

# aquin watch ingest + sae train both call require_active_session, so the
# container needs the CLI token. Create with:
#   modal secret create aquin-token AQUIN_TOKEN=aq-...
AQUIN = modal.Secret.from_name("aquin-token")


# --- Stage 1: ground truth -- CPU, fan out ----------------------------------
@app.function(volumes=VOLS, cpu=2.0, timeout=60 * 60, max_containers=64)
def collect_shard(policy: str, seed: int, episodes: int):
    import subprocess, sys
    subprocess.run([sys.executable, "/root/scripts/collect_data.py",
                    "--policy", policy, "--episodes", str(episodes),
                    "--shard-start", str(seed * 1000)], check=True)
    data_vol.commit()


@app.local_entrypoint()
def stage1_data(episodes_per_shard: int = 256):
    """~8k episodes across 32 CPU containers."""
    jobs = ([("random", s, episodes_per_shard) for s in range(16)] +
            [("ppo", s, episodes_per_shard) for s in range(16)])
    list(collect_shard.starmap(jobs))
    print(f"stage 1 done: {len(jobs) * episodes_per_shard} episodes requested")


# --- Stage 2: world model -- one GPU ----------------------------------------
@app.function(volumes=VOLS, gpu="H100", cpu=16.0, memory=65536,
              timeout=8 * 60 * 60, secrets=[AQUIN])
def train(variant: str = "aux", epochs: int = 12, sigreg_lambda: float = 0.1):
    """
    cpu=16 is not optional: at 224px the loader must decompress ~200 GB/epoch,
    and one core decompresses at ~300 MB/s. Fewer cores idles the H100.
    """
    import subprocess, sys
    subprocess.run([sys.executable, "/root/scripts/train.py",
                    "--variant", variant, "--epochs", str(epochs),
                    "--sigreg-lambda", str(sigreg_lambda)], check=True)
    runs_vol.commit()
    return f"/runs/{variant}_lam{sigreg_lambda}"


@app.local_entrypoint()
def stage2_train(variant: str = "aux", epochs: int = 12):
    print(train.remote(variant=variant, epochs=epochs))


# --- Stage 3: gates + inspect -- cheaper GPU --------------------------------
@app.function(volumes=VOLS, gpu="A10G", cpu=4.0, timeout=2 * 60 * 60, secrets=[AQUIN])
def gates(ckpt: str):
    import subprocess, sys
    subprocess.run([sys.executable, "/root/scripts/run_gates.py", "--ckpt", ckpt], check=True)
    runs_vol.commit()


@app.function(volumes=VOLS, gpu="A10G", cpu=4.0, timeout=2 * 60 * 60, secrets=[AQUIN])
def sae(ckpt: str, tag: str = "base", layer: int = 0):
    """Dump latents -> chunk_*.pt, then aqmario.aq_sae.train_latent_sae."""
    import sys
    sys.path.insert(0, "/root")
    from aqmario.aq_sae import dump_latents, train_latent_sae
    d = dump_latents(ckpt, out_dir=f"/runs/lat/{tag}")
    train_latent_sae(d, tag=tag, layer=layer, output=f"/runs/sae/{tag}.pt")
    runs_vol.commit()


@app.local_entrypoint()
def stage3_gates(ckpt: str):
    gates.remote(ckpt)


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
def plan(ckpt: str, sae_path: str = ""):
    """Emulator + GPU in one container: CEM rolls out in latent space, the
    chosen macro executes on the real env, repeat."""
    import subprocess, sys
    cmd = [sys.executable, "/root/scripts/run_plan.py", "--ckpt", ckpt]
    if sae_path:
        cmd += ["--sae", sae_path]
    subprocess.run(cmd, check=True)
    runs_vol.commit()


@app.local_entrypoint()
def stage4_plan(ckpt: str, sae_path: str = ""):
    plan.remote(ckpt, sae_path)
