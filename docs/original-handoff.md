# AQ-Mario — Build Handoff (for Claude Code)

You are implementing a JEPA world model for Super Mario Bros as an instance of Aquin's interpretability-in-the-loop thesis. Compute runs on **Modal**; the interpretability loop runs on **Aquin / AQIT** (`aq`). This doc is the full context: the problem, the prior work we're beating (with links), the design, the file map, and how to wire Modal and `aq`.

Two standing rules for this build:
- **Verify links before relying on them.** The three primary links in §3 are confirmed. Everything in §11 marked *(verify)* must be fetched and confirmed before use — do not trust a remembered URL or dataset path.
- **Do not invent `aq` CLI syntax.** Consult the AQIT repo README/SDK for exact command and function signatures. Where this doc says "`aquin watch`" or "eval gate," treat it as the intent; get the real invocation from the repo.

---

## 1. Context — Aquin and the thesis

Aquin (aquin.app) is a mechanistic-interpretability company. Thesis: frontier ML lacks the inspect–localize–intervene–verify grip that software engineering takes for granted, and the fix is to move interpretability — SAEs, causal attribution, steering, checkpoint diffs — **out of post-hoc analysis and into the development loop itself**.

The tool is **AQIT**, a CLI + Python SDK structured around a loop:

```
recipe → train → eval gate → inspect on fail → steer/patch
```

`aq` records runs locally (`~/.aquin/commands/`) and can ingest live training metrics (`aquin watch`). Relevant tooling for this project: SAE capture/train/align/diff, linear probes / attribution, residual-drift tracking, weight + feature diffs, and custom eval gates.

**This project is a first-class demonstration of that loop.** Mario is the vehicle; the reusable eval-gate + inspect harness is the product.

---

## 2. The project in one paragraph

Train a compact (~15M param) JEPA world model that predicts Mario's dynamics in a 192-dim latent space (no pixel decoding). Gate **every checkpoint** on interpretability checks that predict planning-readiness *before any planner is trained*. When a gate fails, localize the cause with an SAE on the latent and repair by steering. Then plan with the model to clear World 1-1. Deliverable is two things at once: a working world model whose internals we can verify, and a set of gates that generalize to any action-conditioned world model.

---

## 3. Prior work we're beating (confirmed links)

- **Paper — "LeWorldModel" (the JEPA world-model method):**
  - PDF: https://arxiv.org/pdf/2603.19312
  - Abstract: https://arxiv.org/abs/2603.19312
- **LeMario — Benjamin Bai's Mario implementation of the method (the thing we're beating):**
  - Project writeup: https://www.benjamin-bai.com/projects/lemario
  - Secondary coverage: https://daily.dev/posts/lemario-super-mario-bros-trained-on-a-jepa-model-vlkw5cnrf

**Terminology:** the method is *LeWorldModel* (a JEPA), **not** V-JEPA (that's Meta's video encoder). Get this right in any writeup.

**Bai's published numbers (our targets):**

| Metric | LeMario | Target | Lever |
|---|---|---|---|
| 1-step latent MSE | 0.01377 | < 0.012 | more data, more epochs |
| 5-step latent MSE | 0.07772 | < 0.05 | same |
| 5-step gain over persistence | 45.5% | > 60% | frame-skip 2 |
| x-probe R² | 0.997 | ≥ 0.97 (hold) | don't regress |
| **y-probe R²** | **0.188** | **> 0.90** | skip-2 + jump-heavy data + aux y head |
| scroll-probe R² | (unmeasured) | > 0.90 | aux scroll head (new instrument) |
| Clear 1-1 by planning | never | first time | macro-CEM + probe cost + sub-goals |

The y-probe number is the headline. Bai's encoder barely encoded Mario's height (R²=0.188), so any goal needing a jump was invisible to the planner.

---

## 4. Why LeMario failed → our four fixes

1. **Scrolling-camera aliasing** — places ~1,400px apart look near-identical, so latent distance ≠ progress. → **aux scroll head** forces the encoder to distinguish look-alike locations; **camera-aliasing gate** unit-tests it.
2. **Weak vertical state** (y-R²=0.188) — skip-5 + SIGReg discard height because it barely helps 1-step prediction. → **frame-skip 2** (jump arcs ~30 frames survive) + **aux y head**.
3. **Predictive state ≠ control state** — encoder represents whatever predicts frames (enemy animation, timer, HUD), not what the planner needs. → **aux losses** bend the representation toward controllable state without dominating the dynamics objective.
4. **Data mismatch** — 280 eps / 1 epoch / 32 levels vs the paper's 20k / 10 epochs / 1 env. → **~8k episodes on World 1-1 only, 10+ epochs**.

Keep a `pure` vs `aux` switch (see `config.py`). The diff between the two variants ("what does control-state supervision cost the dynamics?") is itself a publishable result — wire it so both can run and be diffed.

---

## 5. Repo layout & current state

```
aqmario/
  config.py        ✅ WRITTEN — schema, dims, loss weights, gate thresholds, macro-actions
  ram.py           ⬜ shared RAM→state extraction (SEE §10 — define ONCE, import everywhere)
  model.py         ⬜ ViT-Tiny encoder + action-encoder + AdaLN-Zero predictor (~15M)
  losses.py        ⬜ SIGReg + aux heads (y / scroll / alive)
  gates.py         ⬜ probe gate / camera-aliasing gate / dead-in-5 gate
  sae.py           ⬜ SAE train + feature attribution + λ-sweep diff
  plan.py          ⬜ categorical macro-action CEM + probe cost + sub-goals
scripts/
  collect_data.py  ✅ WRITTEN — emulator + RAM logging → .npz shards
  train_ppo.py     ⬜ quick SB3 PPO for the second data track (optional)
  train.py         ⬜ training loop + `aquin watch` hooks
  run_gates.py     ⬜ standalone gate runner on a checkpoint
modal_app.py       ⬜ Modal entrypoints (data-gen, train, gates, sae) — see §7
```

`config.py` already defines: button order (load-bearing — probes and the action-encoder assume it), macro-action set for the planner, data schema, model dims, the aux-loss weights, and the gate thresholds. **Read it first; everything keys off it.**

`collect_data.py` already logs (frame, per-emulator-frame buttons, RAM state) to shards, with the SMB RAM addresses inline. Those addresses are from memory — see §10, item 1.

---

## 6. The AQIT loop, stage by stage (with acceptance criteria)

### Stage 1 — Data (`collect_data.py`, done; `train_ppo.py`, optional)
- One level (`SuperMarioBros-1-1-v0`), frame-skip 2, 224×224 RGB, RAM logged.
- Two policies: right/run/jump-biased random (covers hazards + vertical motion) and a noisy PPO agent (covers real progress). ~4k episodes each.
- **Accept when:** ≥8k episodes sharded to the Volume, and a 60-frame "walk right" trace shows `world_x` climbing monotonically and `scroll` tracking it past mid-screen (RAM validation, §10).

### Stage 2 — Model + Train (`model.py`, `losses.py`, `train.py`)
- Encoder ViT-Tiny (patch 14) → 192-dim `z` via CLS + MLP-**BN** projection (BN matters; a trailing LayerNorm fights SIGReg).
- Action encoder: (skip×6) button block → 192-dim.
- Predictor: 6 transformer blocks, causal over 3-frame history, actions injected via **AdaLN-Zero** (zero-init gates so action effects grow in), head emits next 3 latents.
- Loss (aux variant):
  `MSE(ẑ,z) + 0.1·SIGReg(z) + 0.05·MSE(y_head,z→y) + 0.05·MSE(s_head,z→scroll) + 0.02·BCE(alive_head,z→alive)`
- SIGReg: project `z` onto 1024 random directions, 1-D normality penalty per direction (Cramér–Wold). This is the only anti-collapse mechanism; no EMA/stop-grad needed.
- **`aquin watch` hooks:** stream `pred_loss`, `sigreg_loss`, each aux loss, and live **effective dimension** (participation ratio) of `z`. Two live failure signatures: SIGReg plateaus high (encoder can't reach Gaussian) or pred_loss→0 with eff-dim collapsing (collapsing despite SIGReg; λ too low).
- **Accept when:** both losses move on a single-shard dry run; full run trains 10+ epochs bf16; watch curves are being ingested by `aq`.

### Stage 3 — Eval gate (`gates.py`, `run_gates.py`) — the core Aquin claim
Run per checkpoint; **fail the run** if a gate trips.
- **Probe gate:** freeze encoder, fit MLP probes `z→{x,y,scroll,alive}`. Require y-R²>0.80, x-R²>0.95, scroll-R²>0.90. Use Bai's split protocol (complete-trajectory train/val, ~60 trajectories) so R² is comparable to his.
- **Camera-aliasing gate:** frame pairs from the same level >500px apart must have latent distance > margin. The scrolling-camera bug as a unit test.
- **Dead-in-5 gate:** alive-probe rolled 5 steps forward as a sanity check.
- **Accept when:** gates run standalone on a checkpoint and emit pass/fail + the numbers, and are registered as an AQIT eval gate so a failing checkpoint stops the run automatically.

### Stage 4 — Inspect on fail (`sae.py`)
- Train a small SAE on cached 192-dim latents. Rank features by variance explained for x vs y vs scroll vs enemy-state.
- λ-sweep {0.01, 0.1, 0.5}: diff feature sets; show *which* features λ=0.5 destroys (paper shows λ=0.5 kills planning).
- Optional baseline: feature-purity vs a DINO-WM latent.
- **Accept when:** SAE + attribution runs on a checkpoint's cache and the λ-diff produces a feature-level table.

### Stage 5 — Steer / Plan (`plan.py`)
- Planner: categorical CEM over the 5 macro-actions (`config.MACRO_ACTIONS`), horizon ~8, replan every 2–3 macros.
- Cost: probe-scored — `-x_progress(ẑ) + death_penalty·P(dead) + λ·‖ẑ_y−goal_y‖` — with sub-goals every ~200px so no single latent-distance call spans the aliasing gap.
- Steering demo: take the SAE's y-feature, steer it in the predictor, confirm rollouts jump.
- **Accept when:** nearby-goal sanity plan works, then a sub-goal chain clears 1-1 (the demo GIF).

---

## 7. Running on Modal

Use Modal for both tracks: **data-gen is CPU-parallel** (emulator-bound, the H100 does nothing for it), **training/gates/SAE are GPU**. Persist everything to a Volume so stages compose.

Skeleton (`modal_app.py`) — fill in against current Modal API, verify decorator/Volume syntax against Modal's docs:

```python
import modal

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")  # nes-py / opencv deps
    .pip_install(
        "gym-super-mario-bros==7.4.0", "nes-py", "opencv-python-headless",
        "numpy", "tqdm", "torch", "timm", "stable-baselines3", "aquin",
    )
    .add_local_dir("aqmario", "/root/aqmario")   # mount the package
)
app = modal.App("aq-mario", image=image)
data_vol = modal.Volume.from_name("aqmario-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("aqmario-runs", create_if_missing=True)
VOLS = {"/data": data_vol, "/runs": runs_vol}

# --- Stage 1: data-gen, CPU, fan out across shards -------------------------
@app.function(volumes=VOLS, cpu=2.0, timeout=60*60, max_containers=64)
def collect_shard(policy: str, seed: int, episodes: int):
    # import inside the function; run scripts/collect_data.py logic here,
    # write shard to /data/<policy>/shard_<seed>.npz, then data_vol.commit()
    ...

# --- Stage 2: train, one GPU ----------------------------------------------
@app.function(volumes=VOLS, gpu="H100", timeout=8*60*60)
def train(variant: str = "aux", epochs: int = 12):
    # read shards from /data, write checkpoints to /runs, stream aquin watch
    ...

# --- Stage 3/4: gates + SAE, cheaper GPU ----------------------------------
@app.function(volumes=VOLS, gpu="A10G", timeout=2*60*60)
def gates(ckpt: str): ...
@app.function(volumes=VOLS, gpu="A10G", timeout=2*60*60)
def sae(ckpt: str): ...

@app.local_entrypoint()
def main():
    # parallel data-gen:
    list(collect_shard.starmap(
        [("random", s, 256) for s in range(16)] +
        [("ppo",    s, 256) for s in range(16)]
    ))
    ck = train.remote(variant="aux", epochs=12)
    gates.remote(ck); sae.remote(ck)
```

Notes:
- `.starmap`/`.map` gives you the parallel data-gen; 32 containers × 256 eps ≈ 8k episodes in ~one wall-clock hour.
- Call `.commit()` on the Volume after writing shards/checkpoints so later stages see them.
- Keep the emulator headless (`opencv-python-headless`, no display). nes-py runs fine CPU-only.
- For the `pure` vs `aux` diff, run `train.remote(variant=...)` twice into separate `/runs` subdirs, then diff with `aq`.

---

## 8. Wiring `aq` / AQIT

Get exact commands from the AQIT repo (§11) — below is the intent to implement, not verified syntax.

- **recipe:** express the run (model = `config.ModelConfig`, loss = `config.LossConfig`, data path) as an AQIT recipe so the run is recorded under `~/.aquin/commands/` and uploaded to the aquin.app inbox.
- **train + watch:** in `train.py`, emit metrics through `aquin watch` (pred/sigreg/aux losses + eff-dim) so the curves stream live.
- **eval gate:** register `gates.py` checks as AQIT eval gates keyed to `config.GateConfig` thresholds, so a failing checkpoint halts the run — this is the "catch it at epoch 2" behavior that is the whole point.
- **inspect:** run the SAE capture/train/diff tooling on the cached latents; use feature-diff for the λ-sweep.
- **steer/patch:** use the steering tooling to push the y-feature in the predictor for the planning demo.

If a needed `aq` capability isn't obvious from the repo, stop and flag it rather than faking a flag.

---

## 9. Build order / milestones

1. `ram.py` — shared RAM→state extraction (unblocks everything; §10).
2. Validate RAM addresses locally on a 60-frame trace (10 min, critical).
3. `modal_app.py` data-gen path → kick off ~8k episodes; start PPO on the side.
4. `model.py` + `losses.py` → single-shard dry run, confirm both losses move.
5. `train.py` + `aquin watch` → full run on Modal H100.
6. `gates.py` + `run_gates.py` → per-checkpoint; watch y-R² climb past 0.80.
7. `sae.py` → SAE + λ-sweep diff once a checkpoint passes gates.
8. `plan.py` → nearby goal, then sub-goal chain to the flagpole.
9. Writeup: prediction table, probe table (y 0.188 → >0.90), aliasing gate, SAE λ-diff, 1-1 GIF.

---

## 10. Do-first & gotchas

1. **Validate RAM addresses before collecting 8k episodes.** `collect_data.py`'s SMB addresses are from memory. Step right ~60 frames, print `world_x` and `scroll`: `world_x` should rise ~1–2/frame, `scroll` should track it past mid-screen. A wrong address silently corrupts every gate and probe. Fix in `ram.py`, not in two places.
2. **Define RAM→state extraction ONCE** in `ram.py` and import it into both `collect_data.py` (logging) and `gates.py` (probe targets). If training targets and eval probes drift apart, your numbers are meaningless.
3. **Button order in `config.BUTTONS` is load-bearing** — the action encoder and every probe assume it. Don't reorder.
4. **Frame-skip 2, not 5** — this is a deliberate fix, not a default. Don't "optimize" it back to 5.
5. **BN, not LN, on the encoder projection** — a trailing LayerNorm fights SIGReg's Gaussian pressure.
6. **Don't tie anything to training epoch** as a world property — epoch is a property of the model, not the level.
7. **It's LeWorldModel / JEPA, not V-JEPA** — in code comments and any writeup.
8. **Comparability:** use Bai's probe split protocol or the R² isn't comparable to his 0.188.

---

## 11. References & links

**Confirmed (from this project's conversation):**
- Paper PDF — https://arxiv.org/pdf/2603.19312
- Paper abstract — https://arxiv.org/abs/2603.19312
- LeMario writeup — https://www.benjamin-bai.com/projects/lemario
- LeMario coverage — https://daily.dev/posts/lemario-super-mario-bros-trained-on-a-jepa-model-vlkw5cnrf

**To verify before use (names surfaced earlier; confirm exact URL/path by fetching):**
- `gym-super-mario-bros` (PyPI, Kautenja) — the emulator env. *(package name is stable; pin a known version)*
- `rafaelcp/smbdataset` — human SMB gameplay with per-frame RAM snapshots (Hugging Face). Useful as extra probe targets. *(verify)*
- `lucas-maes/le-wm` — reference LeWorldModel implementation. *(verify)*
- `keon/jepa` — minimal educational JEPA/LeWorldModel implementations. *(verify)*
- Aquin AQIT repo (github.com/Aquinf03/AQIT) — **source of truth for `aq` command/SDK syntax.** *(verify org/repo path)*
- Modal docs — verify current `Image`, `Volume`, `@app.function(gpu=...)`, and `.map`/`.starmap` syntax against https://modal.com/docs

Anything below a confirmed link that you're about to depend on: fetch it first.
