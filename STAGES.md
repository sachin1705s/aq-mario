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

**Two data tracks, not one.** The biased random policy tops out at
`world_x = 1416` of the flagpole's **3161** — measured, not assumed. Everything
past the first pipe complex is simply absent, so a probe R² measured on
random-only data describes the first 45% of 1-1 and is not comparable to a
number measured on the whole level. Half the dataset therefore comes from a
noisy PPO explorer (`scripts/train_ppo.py`), stopped early on purpose at
`--target-x 2400`: a policy that clears the level deterministically collects the
same optimal trajectory 4000 times, which is worse for coverage than one that
dies in interesting places.

`aqmario/ppo.py` holds the grayscale/resize/frame-stack **once**, imported by
both the trainer and `collect_data.py`. The draft called `model.predict()` on
the raw 240×256×3 frame; SB3 would not have raised, the policy would have been
reading noise, and 4000 GPU-funded episodes would have come back
indistinguishable from the free ones.

**Watch the budget.** At 224px / skip-2, 8k episodes is ~6.6M observations and
**992 GB of raw pixels** — but NES art deflates 27×, measured on real shards, so
that is 37 GB on disk and training is not network-bound after all. `shard_size`
dropped 256 → 16 (256 PPO episodes is ~42 GB resident before the first write —
an OOM). The tracker prints this projection.

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

**Both diagnostics are biased by batch size, and both have exact corrections.**
This matters because the λ-sweep compares three runs, and a raw number read
across different batch sizes is meaningless.

*`eff_dim`* (participation ratio, `aqmario/aq_watch.py`). For n samples of a
genuinely full-rank D-dim latent, Marchenko–Pastur gives `E[PR] = nD/(n+D)`, so
a **healthy** encoder at batch 64 (n = 64×6 = 384 latents) reads **128, not
192**. Measured against the formula at D=192: n=64 → 47.1 vs 48.0; n=384 →
127.6 vs 128.0; n=2048 → 175.5 vs 175.5. Inverting it,
`Deff = PR·n/(n−PR)`, recovers 192.0 ± 1 at every n and stays at 7.6 on a rank-8
latent. Raw PR is the number to watch live; the corrected one is the number for
the writeup.

*`sigreg`* has a closed-form null: `E[T | z~N(0,I)] = √π(1−1/√2)/n = 0.51914/n`,
verified within 3% from n=64 to n=2048. So at batch 64 the term **floors at
0.0013** however Gaussian the encoder gets. `sigreg_ratio = sigreg/null` is the
comparable readout, and ~1.0 means "Gaussian at the resolution this batch size
can measure". `losses.py` carries the full table of what the statistic catches
(rank collapse 254× null, wrong scale 321×, anisotropy 10×) and what it does
**not** (per-coordinate bimodality, a shell at the right radius — the projection
CLT hides both). It is a guarantee against collapse, not a certificate of
Gaussianity; do not claim the latter.

**The one experiment that says this design works.** After the BN, a rank-8
latent reads 10× null rather than 254× — BN already fixes the covariance
diagonal, so BN and SIGReg are not redundant, they split the job. And SIGReg
escapes collapse only when the gradient flows *through* the BN, exactly as
`Encoder` is wired:

| step | sigreg | ×null | eff_dim |
|---|---|---|---|
| 0 | 0.0139 | 10.0 | 7.9 |
| 100 | 0.0051 | 3.8 | 21.5 |
| 300 | 0.0021 | 1.6 | 52.0 |
| 600 | 0.0015 | 1.1 | 69.5 |

Pinned as `test_sigreg_escapes_rank_collapse_through_bn`.

**Parameter count is 9.8M, not 15M** (encoder 5.65M + predictor 4.04M +
action-encoder 0.04M + aux heads 0.07M) with `predictor_layers: 6` as specified
above. The tracker reports this as under-band rather than the band being widened
to fit; closing the gap means raising `predictor_layers`, which is a deliberate
config change and an experiment, not a rounding.

### The collapse, and what it took to find it

The first real dry run **passed the gate while collapsing**, and unpicking that
produced most of what is now known about this model. In order:

**1. The gate had a hole.** It asserted "pred_loss moved and sigreg_loss moved".
A collapsing run satisfies both — `pred_loss 0.975 -> 0.061`, `sigreg 0.030 ->
0.098`, `eff_dim 9.96 -> 1.17`. The prediction task is trivially solvable by
mapping every frame to one point, so a falling loss is evidence of nothing on
its own. The gate is now a pure function (`train.dryrun_verdict`) with that exact
history pinned as a regression test.

**2. `gain` cannot be quoted without `eff_dim`.** The encoder ends in
BatchNorm(affine=False), so var(z) = 1 per coordinate *by construction* and a
mean-predictor scores exactly 1.0 — which is what makes `gain = 1 - mse`
comparable to LeMario at all. But a rank-1 latent ALSO has unit variance per
coordinate (every coordinate perfectly correlated), so the predictor only has to
predict one scalar. That collapsed run scored `gain_5step = 0.939` against
LeMario's 0.455. Quote the two together or not at all.

**3. Both diagnostics were being computed on the wrong sample count.** A batch is
B windows x W near-identical frames, so it has B independent samples, not B*W.
Scoring a *healthy* latent against the n=B*W null reports 6.7x null and eff_dim
27 against an apparent null of 96 — a fake 3.5x collapse. Worse, flattened SIGReg
penalises the within-window similarity the predictor depends on. Both are now
computed per frame position and averaged (`sigreg_windowed`,
`effective_dim_windowed`), which keeps every latent and makes the null exact.

**4. SIGReg's power is a function of batch size, and ours was far too small.**
Ratio to null for a rank-r latent after BN:

| rank | n=32 | n=64 | n=128 | n=512 | n=2048 |
|---|---|---|---|---|---|
| 2 | 2.8× | 5.6× | 10.9× | 41.8× | 173.8× |
| 4 | 1.8× | 3.3× | 6.4× | 25.0× | 97.8× |
| 16 | 0.8× | 1.2× | 2.0× | 7.2× | 28.5× |
| 192 (healthy) | 0.4× | 0.5× | 0.5× | 1.0× | 2.8× |

At batch 32 a rank-2 latent reads 2.8× against a healthy 0.4× — barely
separable, so the term cannot push back on a collapse it can hardly see. This
puts SIGReg and the memory budget in direct opposition: a step encodes
`batch x window` images (8 frames per window at horizon 5), and batch 128 OOMs a
24 GB A10G. Hence `grad_checkpointing: True` — it is what buys the batch.

**5. λ alone cannot fix it, and the arithmetic says why.** Collapsing is worth
~0.80 of `pred_loss`. Against that:

| λ | pred saved | sigreg cost | aux cost | ratio |
|---|---|---|---|---|
| 0.1 | 0.796 | 0.010 | 0.008 | **45×** |
| 1.0 | 0.778 | 0.065 | 0.008 | **11×** |
| 5.0 | 0.745 | 0.181 | 0.008 | **4×** |

Every anti-collapse term combined is one to two orders of magnitude smaller than
the incentive. That is not a tuning problem. Measured eff_dim across that sweep:
1.08 → 1.89 → 3.50 for λ = 0.1 → 1 → 5, all collapsed.

**6. The stop-gradient is what actually moves it.** With a shared encoder and no
stop-grad, `MSE(ẑ, z_target)` is minimised by making `z_target` constant — the
target path is trainable, so collapse is not a side effect, it is the direct
solution. LeJEPA's claim is that SIGReg makes the stop-grad unnecessary; at our
batch size and step count, measured, it does not:

| batch | stop-grad | eff_dim (init → 500 steps) | gain_5 |
|---|---|---|---|
| 32 | ✗ | 7.9 → **2.0** | 0.969 *(fake)* |
| 128 | ✗ | 11.1 → **2.0** | 0.978 *(fake)* |
| 32 | ✓ | 7.9 → **5.8** | 0.624 |
| 128 | ✓ | 11.1 → **8.4** | **0.651** |

`gain_5step = 0.651` against LeMario's 0.455 — and this one is real, because
eff_dim held. This is a documented departure from the plan's "no stop-gradient",
kept as an explicit ablation rather than a silent default.

**7. What actually works.** With the stop-gradient on, λ becomes an effective
lever — and at λ=10 the effective dimension *doubles* over training rather than
falling. 1500 steps, batch 96, all four cells pass:

| λ | aux_scale | eff_dim (init → end) | gain_5 | sigreg |
|---|---|---|---|---|
| **10** | **20** | **10.3 → 21.1** | 0.654 | 1.4× |
| 10 | 1 | 10.3 → 20.4 | 0.545 | 1.5× |
| 1 | 20 | 10.3 → 9.7 | 0.778 | 3.0× |
| 1 | 1 | 10.3 → 8.9 | 0.817 | 3.3× |

There is a real trade-off in that table: λ=1 predicts better (gain 0.82) on a
narrower latent, λ=10 holds a much wider latent for gain 0.65. Both beat
LeMario's 0.455 — but which one is *right* is decided by the Stage 3 y-probe,
not by either column here, which is the entire argument for gating on the
representation instead of the loss.

`aux_scale` exists because the original aux weights were ~100× too weak to
influence anything (0.008 of cost against 0.80 of incentive). The Modal
defaults are now λ=10, aux_scale=20, stop_grad=True.

**8. The collapse threshold itself was miscalibrated.** An *untrained* encoder
starts at eff_dim 7.9 against an MP null of 27 (batch 32) — every Mario frame is
visually similar, so a random ViT is already near-degenerate. A "> 0.5 × null"
bar demands more than initialisation provides and no run could pass it. The check
is now relative to init: rank must not get worse, and must not be rank-1.

---

**AdaLN-Zero has a one-step blind spot.** With the modulation weights at exactly
zero, `d(loss)/d(action_code)` is also exactly zero, so the action encoder gets
no gradient on step 0. The AdaLN weights themselves do (the gate multiplies a
non-zero branch), so the path unsticks on step 1. Pinned as a test, because if
it did not unstick the action encoder would be dead for the whole run and this
would be a video predictor.

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

### First measured gate run

Checkpoint: 4,000 steps (≈0.55 of one epoch, ~4% of the planned schedule), batch
96, λ=10, aux_scale 20, stop-grad on, random-policy data only. Bai's protocol:
60 complete held-out trajectories, 24,640 latents.

| probe | this run | LeMario | threshold | |
|---|---|---|---|---|
| **y** | **0.943** | **0.188** | 0.80 | **PASS — 5.0×** |
| x | 0.963 | 0.997 | 0.95 | PASS |
| scroll | 0.962 | — | 0.90 | PASS |
| dead-in-5 | AUC 0.967 / AP 0.252 | — | — | 31× the 0.008 base rate |
| aliasing (strict margin) | −1.08 | — | 0.10 | **FAIL** |

The one thing this project exists to move went from 0.188 to 0.943, on 4% of the
training budget. It was not a hard problem once it was being measured — which is
the whole argument.

### The result above is confounded, and the control says so

The model measured above was trained with `variant="aux"` and `aux_scale=20`,
which puts an explicit `MSE(y_head(z), y)` term in the loss. **It was trained to
make y decodable, and then measured for whether y is decodable.** LeMario had no
such term, so the comparison as first written was not a comparison.

The control is the `pure` variant — identical recipe, identical data, identical
step count, aux terms removed from the loss entirely. Both gated on the same full
8k dataset and the same 60 held-out trajectories:

| probe | aux (y supervised) | **pure (no supervision)** | LeMario |
|---|---|---|---|
| **y** | 0.796 | **−0.383** | 0.188 |
| x | 0.870 | 0.451 | 0.997 |
| scroll | 0.878 | 0.464 | — |
| dead-in-5 AUC | 0.927 | 0.628 | — |
| aliasing usable range | 350 px | 0 px | — |

**A pure JEPA on this recipe does not discover Mario's height at all.** −0.383 is
worse than predicting the mean, and worse than the number this project set out to
beat. Every point of the 0.796 came from the auxiliary supervision.

So the claim "this JEPA learns a better representation than LeMario's" is
**false** and should not be made. What the evidence supports:

1. **Pure JEPA reproduces LeMario's failure mode, and then some.** That is
   corroboration of their finding, not a refutation of it — at equal budget, and
   at 4% of a full schedule.
2. **A cheap auxiliary head fixes it** — 0.796 against 0.188 — at a cost of
   0.07M parameters and one loss term.
3. **The gate caught the pure variant's failure at 4,000 steps**, roughly 35
   minutes, with no planner built. That is the actual thesis of the project and
   it is the part that held up.

Caveat kept in view: 4,000 steps is 4% of the schedule and self-supervised
objectives are slower to develop than supervised ones. Both variants are
therefore being run to the full 6 epochs on the full 8k dataset, and both numbers
will be reported. If pure catches up, point 1 above weakens and should be
rewritten; if it does not, the honest headline for this project is point 2 plus
point 3, not a representation-quality win.

**The aliasing gate fails, and unrolling it is more useful than the number.**
Mean latent distance by world-x separation:

| \|Δx\| px | 0-20 | 20-50 | 50-100 | 100-200 | 200-350 | 350-500 | 500-800 | 800-1500 |
|---|---|---|---|---|---|---|---|---|
| mean dist | 11.32 | 12.79 | 15.66 | 15.95 | 17.03 | **16.50** | **15.87** | 17.12 |

Distance rises cleanly to ~350 px and then dips. The NES screen is 256 px wide,
so that is the camera: two frames about two screens apart start to look alike.
Spearman over all pairs is +0.453.

The strict `aliasing_margin` compares the 5th percentile of far-pair distance to
the 95th percentile of near-pair distance, i.e. it demands the two distributions
barely overlap — and it is measuring the wrong thing here for two reasons.
"Near" means near in **x only**: Mario standing, mid-jump and dying all sit at
one x, so near-pair distance has a floor of 11.3 that has nothing to do with the
camera. And decisively, **x is recoverable from this latent at R² 0.963** — a
latent you can read x out of to within 4% of its variance is not aliased in the
sense the margin claims.

So the gating criterion is now `aliasing_usable_range`: the largest separation
out to which mean distance is still non-decreasing. **Measured: 350 px.** The
planner's `subgoal_px` is 200, chosen on the plan's intuition before any of this
existed. The guess and the measurement agree, and the sub-goal design is now safe
for a stated reason rather than a plausible one. The strict margin is still
computed and still reported as failing (`aliasing_margin_strict_fails`), because
a threshold quietly retired is a threshold that was never a threshold.

---

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
