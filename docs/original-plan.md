# AQ-Mario: Plan

A JEPA world model for Super Mario Bros, built as the AQIT loop, with the interp gates as the headline result. Aux-loss variant (y/scroll from RAM). Win conditions for today: beat LeMario's prediction + probe numbers, clear World 1-1 by planning, and ship the eval gates + an SAE finding. Training on a rented H100/A100.

The through-line: **a model that predicts perfectly can still fail to plan, and the interp gate finds that failure before you waste a training run.** That claim is the deliverable. Mario is the vehicle.

---

## Targets to beat (all from Benjamin Bai's LeMario postmortem)

| Metric | LeMario | Your target | Why it's reachable |
|---|---|---|---|
| 1-step latent MSE | 0.01377 | < 0.012 | 30x more data, 10 epochs vs 1 |
| 5-step latent MSE | 0.07772 | < 0.05 | same |
| 5-step gain over persistence | 45.5% | > 60% | shorter frame skip makes actions matter |
| x-probe R² | 0.997 | ≥ 0.97 (hold) | already strong; don't regress |
| **y-probe R²** | **0.188** | **> 0.90** | the headline fix: skip-2 + jump-heavy data + aux head |
| scroll-probe R² | (not measured) | > 0.90 | new instrument; guards camera aliasing |
| Clear 1-1 by planning | never | first time | macro-action CEM + probe cost + sub-goals |

The y-probe number is the whole story. LeMario's encoder barely knew Mario's height, so any goal requiring a jump was invisible to the planner. Three of his failure modes collapse into "fix vertical state."

---

## Why LeMario failed (so we don't repeat it)

1. **Scrolling camera aliasing.** Two spots ~1,400px apart look nearly identical, so latent distance ≠ progress. Push-T has a fixed camera; the method's core assumption (visual similarity = progress) silently broke.
2. **Weak vertical state.** y-R² = 0.188. Frame-skip 5 + SIGReg happily discards height because it barely helps 1-step prediction.
3. **Predictive state ≠ control state.** The encoder represents whatever predicts the next frame (enemy animation, timer, HUD). The planner needs a space where distance = controllable progress. CEM exploits every gap between the two.
4. **Data mismatch.** 280 episodes / 1 epoch / 32 levels, vs the paper's 20k / 10 epochs / 1 environment. He imported the architecture and dropped the conditions that made it work.

Our design decisions map 1:1 onto these: aux scroll head → (1), aux y head + skip-2 → (2), aux losses in general → (3), 8k episodes on 1-1 only → (4).

---

## The AQIT loop, stage by stage

### Stage 1 — Data
`scripts/collect_data.py` (written). `gym-super-mario-bros`, log (frame, 6-bit buttons per emulator-frame, RAM state) to `.npz` shards.

- **One level only** (`SuperMarioBros-1-1-v0`) until it works. Generalization is a later problem.
- **Frame skip 2**, not 5. Jump arcs survive; y becomes prediction-relevant.
- **Two policies:** random (biased right/run/jump — covers hazards and vertical motion), and a noisy PPO agent (covers actual progress). ~4k episodes each.
- **RAM is ground truth** for x, y, scroll, alive, power. Every gate and probe depends on it.

**First thing to do locally — validate RAM addresses.** The addresses in `collect_data.py` are from memory of the SMB RAM map. Before collecting 8k episodes, run: step right ~60 frames, print `world_x` and `scroll`. `world_x` should climb ~1–2/frame; `scroll` should track it once Mario passes mid-screen. If not, fix the address. A wrong address silently corrupts every downstream number.

Throughput note: the H100 does nothing for data-gen (it's CPU/emulator-bound). Run collection in parallel processes; ~8k episodes is a couple of hours wall-clock, not GPU time.

### Stage 2 — Recipe → Train
`aqmario/model.py`, `aqmario/losses.py`, `scripts/train.py`.

**Model (≈15M params):**
- Encoder: ViT-Tiny, patch 14, frame → 192-dim latent `z` (CLS + MLP-BN projection; the BN matters — final ViT LayerNorm otherwise fights SIGReg).
- Action encoder: (skip × 6) button block → 192-dim vector.
- Predictor: 6 transformer blocks, causal over a 3-frame history, actions injected via **AdaLN-Zero** (shift/scale/gate per branch, zero-init so action effects grow in gradually). Projection head emits the next 3 latents.

**Loss (aux variant):**
```
L = MSE(ẑ, z)                    # dynamics
  + 0.1  · SIGReg(z)             # anti-collapse (project to 1024 dirs, Epps–Pulley normality)
  + 0.05 · MSE(y_head(z),  y)    # vertical position   ← repairs failure #2
  + 0.05 · MSE(s_head(z),  scroll) # camera offset      ← repairs failure #1
  + 0.02 · BCE(alive_head(z), alive) # death signal
```
Aux weights are deliberately small: they *bend* the representation toward control-usefulness without letting supervision dominate the JEPA objective. Keep a `variant="pure"` switch so you can run both and diff — that diff is itself a publishable result ("what does adding control-state supervision cost the dynamics?").

**Watch hooks (`aquin watch`):** ingest `pred_loss`, `sigreg_loss`, each aux loss, and a live **effective dimension** of `z` (participation ratio over a held-out batch). Two failure signatures to catch live:
- SIGReg plateaus high → encoder can't reach Gaussian (the TwoRoom pathology; low-dim data fighting a 192-d prior).
- pred_loss → ~0 fast while eff-dim collapses → you're collapsing despite SIGReg; λ too low.

Training budget: 10+ epochs on ~8k episodes of 1-1. On an H100 this is hours and you'll be I/O-bound, not compute-bound. Use bf16, big batch (256+), and prefetch shards.

### Stage 3 — Eval gate (per checkpoint)
`aqmario/gates.py`. This is the part that turns a blog-post postmortem into tooling. Run every N steps; **fail the run** if a gate trips.

- **Probe gate:** freeze encoder, train tiny MLP probes z→{x, y, scroll, alive}. Require y-R² > 0.80, x-R² > 0.95, scroll-R² > 0.90. Use LeMario's split protocol (60 trajectories, complete-trajectory train/val split) so your R² is comparable to his.
- **Camera-aliasing gate:** sample frame pairs from the same level > 500px apart; require mean latent distance > margin. This is the scrolling-camera bug written as a unit test — the single check that most directly predicts whether raw-latent planning will lie.
- **Dead-in-5 gate:** the alive probe rolled forward 5 steps; a sanity check that the model represents imminent death.

The point: LeMario found the y-problem *after* planning failed, by hand. Here it fails the run at epoch 2, automatically. That's the tooling claim.

### Stage 4 — Inspect on fail
`aqmario/sae.py`. When a gate trips (or just to characterize a passing run), train a small SAE on the 192-dim latent over a cached activation set. Rank features by variance explained for x vs y vs enemy-state vs scroll.

Two questions worth a writeup:
- Does SIGReg's Gaussian pressure produce monosemantic axes, or smear features? Compare feature purity to a DINO-WM latent baseline.
- Sweep λ ∈ {0.01, 0.1, 0.5} and diff feature sets. The paper shows λ=0.5 kills planning; show *which* features it destroys.

This is the step with no prior art in this line of work. It's where Aquin's tooling has no competition.

### Stage 5 — Steer / Plan
`aqmario/plan.py`. Two things here.

- **Planner:** categorical CEM over the 5 macro-actions (`wait`, `run_right`, `run_right_jump`, `jump`, `run_left`), horizon ~8, replan every 2–3 macros. Gaussian CEM over 6 binary buttons was half of LeMario's planning failures — the optimizer fighting the action space.
- **Cost:** probe-scored, not raw latent distance. `cost = -x_progress(ẑ) + death_penalty·P(dead) + λ·‖ẑ_y − goal_y‖`. Use sub-goals every ~200px so no single latent-distance call has to span the aliasing gap.
- **Steering demo (the Aquin flourish):** once the SAE gives you a y-feature, steer it in the predictor and confirm the rollout jumps. Feature steering as a planner primitive — nobody in the playable-world-model line has shown this.

**1-1 clear** = chain sub-goals from start to flagpole under the macro-CEM + probe cost. That's the demo GIF.

---

## File map

```
aqmario/
  config.py        ✅ schema, dims, loss weights, gate thresholds
  model.py         ⬜ encoder + action-encoder + AdaLN-Zero predictor
  losses.py        ⬜ SIGReg + aux heads
  gates.py         ⬜ probe / aliasing / dead-in-5 gates
  sae.py           ⬜ SAE + feature attribution
  plan.py          ⬜ macro-action CEM + probe cost + sub-goals
scripts/
  collect_data.py  ✅ emulator + RAM logging → shards
  train_ppo.py     ⬜ quick SB3 PPO for the data track (optional)
  train.py         ⬜ training loop + watch hooks
  run_gates.py     ⬜ standalone gate runner on a checkpoint
configs/           ⬜ hydra/yaml overrides if you want them
```

## Day sequence

1. Validate RAM addresses (10 min, critical).
2. Kick off random-policy collection in parallel; start PPO training on the side for the second data track.
3. While data collects: finish `model.py` + `losses.py`, dry-run on a single shard to confirm shapes and that both losses move.
4. Full train, `aquin watch` on the five curves + eff-dim.
5. Gates every checkpoint. If y-R² gate trips, that's expected early — watch it climb past 0.80.
6. SAE once a checkpoint passes gates; rank features, run the λ-sweep diff.
7. Planner: nearby goal first (sanity), then sub-goal chain to the flagpole.
8. Write up: prediction table, probe table (with the y 0.188 → >0.90 line), aliasing gate, the SAE λ-diff, and the 1-1 GIF.

## Ordering / dependency notes

- Everything downstream reads the RAM state, so a wrong address poisons silently. Validate first.
- The aux heads share the frozen-probe target definitions with the gates — define x/y/scroll extraction **once** (in `config.py` / a shared `ram.py`) so training targets and eval probes can't drift apart.
- Keep `pure` vs `aux` a one-flag switch. The diff between them is a second paper, not a second project.
- Don't tie difficulty or anything else to training epoch — that came up earlier; epoch is a property of the model, not the world.

## What "credibility" actually comes from here

Not "beat a blog post" and not "playable Mario." It's the gates: an automated check that catches a world model's planning failure from its representation alone, before a planner is ever trained. That generalizes past Mario, it's the kind of result LAION/KAIST would co-author, and the DIAMOND playable demo (project #2) becomes the trailer you release on the same data pipeline afterward.
