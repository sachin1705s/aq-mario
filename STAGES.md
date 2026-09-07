# AQ-Mario — four stages

Each stage has one deliverable and one exit criterion. Progress is not
hand-ticked: `python -m aqmario.tracker` probes real state — files, shard
counts, `metrics.jsonl`, gate JSON — and reports what it finds.

```
python -m aqmario.tracker            # the board
python -m aqmario.tracker --stage 2  # one stage
python -m aqmario.tracker --json     # machine-readable
python -m aqmario.tracker --write    # refresh STATUS.md
```

---

## Stage 1 — Ground truth
**Deliverable:** RAM extraction you can trust, and 8k episodes of 1-1 behind it.
**Exit:** RAM addresses locked by a real trace **and** ≥8k episodes sharded.

`aqmario/ram.py` is the single RAM→state definition, imported by both
`collect_data.py` (training targets) and `gates.py` (probe targets). If those
two ever drift apart every number in the writeup is meaningless, so there is
exactly one definition.

It ships **candidate** addresses rather than constants. A wrong address does not
crash — it silently corrupts every gate, probe and planner cost. So each field
declares the candidates plausible in the datacrystal map plus an invariant a
real walk-right trace must satisfy, and `scripts/validate_ram.py` picks the
winner empirically. Until it does, `ram.validated()` is `False` and
`collect_data.py` refuses to run.

```
python scripts/validate_ram.py --lock      # 10 min, blocks everything else
modal run modal_app.py::stage1_data
```

**Watch the budget.** At 224px / skip-2, 8k episodes is ~5.2M observations and
**783 GB of raw pixels**. That is within Modal's free 1 TiB volume tier once
compressed (~196 GB), but it makes training I/O-bound and it is why
`shard_size` dropped 256 → 16 (256 PPO episodes is ~42 GB resident before the
first write — an OOM). The tracker prints this projection.

---

## Stage 2 — World model
**Deliverable:** a ~15M JEPA that predicts 1-1 dynamics in 192-dim latent space.
**Exit:** 10+ epochs bf16, both losses moving, curves live in `aquin watch`.

Encoder ViT-Tiny patch-14 → 192-dim `z` via CLS + MLP-**BN** projection (BN, not
LN — a trailing LayerNorm fights SIGReg). Action encoder (skip×6) → 192.
Predictor: 6 blocks, causal over 3 frames, actions injected via **AdaLN-Zero**.

```
L = MSE(ẑ,z) + 0.1·SIGReg(z)
  + 0.05·MSE(y_head,y) + 0.05·MSE(s_head,scroll) + 0.02·BCE(alive_head,alive)
```

Two live failure signatures, both visible in the watch curves:
- `sigreg_loss` plateaus high → encoder can't reach Gaussian.
- `pred_loss` → 0 while `eff_dim` collapses → collapsing despite SIGReg, λ too low.

`eff_dim` (participation ratio) is in `aqmario/aq_watch.py` and is verified:
183 on full-rank 192-d noise, 8.0 on a rank-8 latent.

---

## Stage 3 — Gates + inspect
**Deliverable:** the Aquin claim — catch a planning failure from the
representation alone, before a planner exists.
**Exit:** gates run per checkpoint, y-probe R² > 0.80, λ-diff table produced.

- **Probe gate** — freeze encoder, fit MLP probes `z→{x,y,scroll,alive}`.
  Require y-R²>0.80, x-R²>0.95, scroll-R²>0.90. Use Bai's split protocol
  (complete-trajectory split, ~60 trajectories) or the R² isn't comparable.
- **Camera-aliasing gate** — frame pairs >500px apart must have latent distance
  above margin. The scrolling-camera bug as a unit test.
- **Dead-in-5 gate** — alive-probe rolled 5 steps forward.

LeMario found the y-problem *after* planning failed, by hand. Here it fails the
run at epoch 2, automatically.

**The λ-sweep is three full training runs, not three SAE refits.** λ is
`sigreg_lambda`, so {0.01, 0.1, 0.5} ≈ 3 × 3.2 h H100 ≈ $50. Modal runs them in
parallel, so it costs money but no extra wall clock.

---

## Stage 4 — Control
**Deliverable:** plan with the model, steer it with a feature the SAE found.
**Exit:** sub-goal chain clears World 1-1; y-feature steering changes the rollout.

- **Planner:** categorical CEM over the 5 macro-actions, horizon ~8, replan
  every 2–3 macros. Gaussian CEM over 6 binary buttons was half of LeMario's
  planning failures — the optimizer fighting the action space.
- **Cost:** probe-scored, `-x_progress(ẑ) + death_penalty·P(dead) + λ·‖ẑ_y−goal_y‖`,
  with sub-goals every ~200px so no single latent-distance call spans the
  aliasing gap.
- **Steering:** push the SAE's y-feature in the predictor, confirm rollouts jump.
  `aquin steer` is prompt-in/tokens-out and cannot do this — it's ours to write,
  about 20 lines adding `α·decoder_col[y_feat]` into the predictor residual.

Flagpole is `world_x ≈ 3161`; the tracker reports progress as a fraction of it.

---

## How `aq` and `aquin` are actually used

Two separate binaries. Verified against the installed source, not the handoff's
assumptions.

**`aq`** (bash → node, "a train is a directory") is the experiment spine. This
repo *is* an aq train.

| Use | Mechanism |
|---|---|
| Run the JEPA under `aq train` | `recipe.yaml` `method: jepa` → **`methods/jepa.py` in this train**. `protocol/method.py` checks `train/methods/` **before** `kernel/methods/`, so a custom method is first-class. |
| The gate, failing closed | `recipe.yaml` `eval.min_score: 1.0` + `methods/jepa.py:evaluate()`. `engine/step.py:do_eval` computes `all_pass` against it. Native. |
| Live curves | kernel writes `artifacts/metrics.jsonl` (`protocol/metrics.py`) |
| NaN / loss blow-up | `recipe.yaml` `guard.safety: true` → raises `GuardAbort` |
| pure vs aux | `aq fork` the train, run both, `aq diff <a> <b>` |
| λ-sweep orchestration | `aq schedule` (sweeps / cron / resume-on-fail) |

Constraint: `do_eval` does `json.loads(ckpt.read_text())`, so a checkpoint must
be JSON. `fit()` writes weights to a `.pt` and returns a pointer.

**`aquin`** (pip v3.0.1) is LLM-scoped almost everywhere — its own help sections
read "Inspection · **LLM**", "Evals · **LLM**". Exactly two things are
model-agnostic enough for a 192-dim JEPA:

| Use | Mechanism |
|---|---|
| Live curves in the Aquin UI | `aquin watch init` → `watch ingest --file metrics.jsonl --follow`. Pure external-metrics observer; knows nothing about architecture. |
| SAE + λ feature diff | Activation store is dimension-agnostic: `chunk_*.pt` of shape `(N,192)`, optional `norm.pt`, **no manifest** (missing manifest short-circuits validation; only a *layer* mismatch raises). Then `aquin sae align`. |

⚠️ The `aquin sae train` **CLI cannot be used** — `cmd_sae_train` never passes
`d_model` down, so the SAE is built at the session model's width (768 for GPT-2)
and silently mismatches 192. `aqmario/aq_sae.py` calls
`aquin.compute.sae_train.train_sae(..., d_model=192, n_features=2048)` directly,
which is the only path that expresses this.

**Not used, deliberately:** `inspect`, `steer`, `feature-logits`, `attention`,
`layer-analysis`, `perturbation`, `check-weights`, `audit`, `red-team`,
`consistency`/`suppression`/`boundary-eval`, `aquin eval`, `simulate` — all
prompt-and-tokenizer shaped. `weight-diff` / `residual-drift` /
`trajectory-analysis` resolve the model from the active session and expect HF
layer naming; **unverified against a ViT-JEPA state_dict, assume no.**

**Neither tool provides compute.** `aq job run --gpu N` is a *local* subprocess
runner (only `subprocess` in `aquin/compute/`, no cloud provider). Modal is
100% of the bill; `aq`/`aquin` are the record-and-inspect layer around it.
Both `watch ingest` and the SAE path call `require_active_session`, so a
container needs `aquin login` + an `AQUIN_TOKEN` secret.
