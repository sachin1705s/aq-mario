# AQ-Mario

**An implementation of [LeJEPA](paper/) (Balestriero & LeCun, arXiv:2511.08544) for an
action-conditioned world model — and a study of what its SIGReg objective needs in that setting.**

> LeJEPA identifies the isotropic Gaussian as the optimal embedding distribution for a JEPA and
> introduces **SIGReg** to enforce it, arguing that this removes the need for anti-collapse
> heuristics. This repository takes SIGReg out of image self-supervised learning and into a world
> model over video with actions, evaluated by whether a *control-relevant state variable* survives
> in the representation. **Start with [`paper/`](paper/)** — it holds the paper, its licence, and a
> claim-by-claim map of where our measurements agree and disagree.

A 9.8M-parameter JEPA trained on 8,000 episodes of Super Mario Bros 1-1 (4.88M observations),
predicting 5 steps ahead in a 192-dimensional latent space.

📄 **[Complete report (14 pages)](AQ-Mario-complete-report.pdf)** ·
📄 [SIGReg study only](SIGReg-world-model-implementation-study.pdf) ·
📄 [Non-technical explainer](AQ-Mario-experiment-report.pdf)

---

## Findings

Six results, in decreasing order of how much they depend on our training budget.

**1. SIGReg has a closed-form null, and without it λ is not comparable across runs.**
Under the null, `E|φₙ − φ|² = (1 − e^{−t²})/n` exactly, so

```
E[T | z ~ N(0, I)] = √π · (1 − 1/√2) / n  =  0.51914 / n
```

Verified within 3% from n=64 to n=2048. At batch 64 the term **floors at 0.0013** however Gaussian
the encoder becomes — so raw SIGReg values from different batch sizes are not comparable, and
neither are their λ. Report `T / null(n)`.

**2. Its power to detect collapse scales with batch size.** Ratio to null for a rank-*r* latent
after BatchNorm, D=192:

| rank | n=32 | n=64 | n=128 | n=512 | n=2048 |
|---|---|---|---|---|---|
| 2 | 2.8× | 5.6× | 10.9× | 41.8× | 173.8× |
| 16 | 0.8× | 1.2× | 2.0× | 7.2× | 28.5× |
| 192 (healthy) | 0.4× | 0.5× | 0.5× | 1.0× | 2.8× |

At n=32 a collapsed latent is barely separable from a healthy one, so the term cannot resist what it
cannot see. A world-model step encodes `batch × window` images, so memory pushes the batch *down*,
into exactly that regime. Activation checkpointing buys statistical power, not just throughput.

**3. A trailing BatchNorm cuts the collapse signal 25×.** The projection head must end in BatchNorm
(a trailing LayerNorm puts embeddings on a fixed-radius sphere and fights the Gaussian target), but
BN pins the covariance diagonal for free: a rank-8 latent reads **254× null raw, 10× after BN**.

**4. Correlated video batches break the iid null.** A batch is `B` windows × `W` near-identical
frames, so it holds `B` independent samples, not `B×W`. Scored naively, a *healthy* latent reads
6.7× null and looks collapsed. Compute the statistic per frame position and average.

**5. The stop-gradient was still required.** 500 steps, identical otherwise:

| batch | stop-grad | effective dim (init → 500) | gain₅ |
|---|---|---|---|
| 32 | ✗ | 7.9 → **2.0** | 0.969 *(fake)* |
| 128 | ✗ | 11.1 → **2.0** | 0.978 *(fake)* |
| 128 | ✓ | 11.1 → **8.4** | 0.651 |

λ alone could not substitute: collapsing buys the optimiser 0.80 of prediction loss, while λ=5's
SIGReg term costs 0.18. With a shared encoder, `MSE(ẑ, z_target)` is *directly* minimised by making
the target constant — collapse is the objective's global optimum unless something removes the
incentive.

**6. Preventing collapse did not preserve usable state.** The SIGReg-only model did **not** collapse
and still could not report the character's height:

| probe (frozen encoder, 60 held-out trajectories) | SIGReg only | + aux heads (0.07M params) |
|---|---|---|
| **height (y)** | **−0.383** | **0.796** |
| position (x) | 0.451 | 0.870 |
| camera (scroll) | 0.464 | 0.878 |
| death within 5 steps (AUC) | 0.628 | 0.927 |

An isotropic-Gaussian embedding is a **floor, not a sufficient condition**. Which is the argument for
gating on a probe of the state the task needs, per checkpoint — it costs minutes and is the only
measurement here a collapsed-but-confident model cannot fake.

> **Read the limitations before citing any of this.** All numbers come from a 4,000-step run
> (~4% of the planned schedule) on one level of one game. Finding 6's aux column is *trained* to make
> height decodable and then measured for it — it shows a cheap head fixes the problem, not that the
> objective learned a better representation. See [report §15](AQ-Mario-complete-report.pdf).

---

## Quick start

```bash
pip install -r requirements.txt
python -m aqmario.tracker          # the stage board, probed from real state
pytest -q                          # 161 checks
```

Nothing in this repository is hand-ticked: `aqmario/tracker.py` derives progress from files on disk,
shard counts, `metrics.jsonl` contents and gate JSON.

## Reproducing

Compute runs on [Modal](https://modal.com); the code has no other cloud dependency.

```bash
python scripts/validate_ram.py --lock          # empirically pick the RAM addresses (10 min)
modal run modal_app.py::preflight              # verify the pinned emulator stack
modal run modal_app.py::stage1_data  --shards 16 --episodes-per-shard 250   # 4k random episodes
modal run modal_app.py::stage1_ppo   --steps 4000000 --target-x 2400        # train the explorer
modal run modal_app.py::stage1_data_ppo --shards 16 --episodes-per-shard 250

modal run modal_app.py::stage2_train --variant aux  --epochs 3   # ~1.8 h H100
modal run modal_app.py::stage2_train --variant pure --epochs 3   # the control

modal run modal_app.py::stage3_gates --ckpt /runs/<tag>/jepa.pt
modal run modal_app.py::stage4_plan  --ckpt /runs/<tag>/jepa.pt
```

Every setting lives in [`recipe.yaml`](recipe.yaml) with the measurement that justifies it, and a
test asserts the recipe never drifts from the code.

## Layout

```
paper/            the LeJEPA paper, its licence, and the claim-by-claim map   <- start here
aqmario/
  config.py       every hyperparameter, each with the measurement behind it
  data.py         shards -> training windows (action alignment, trajectory splits)
  model.py        ViT-Tiny encoder -> BN projection; AdaLN-Zero causal predictor
  losses.py       SIGReg (Epps-Pulley + Gauss-Hermite), its null, aux heads
  train.py        training loop, per-10% checkpoints, fail-closed gate
  gates.py        frozen-encoder probes, camera-aliasing curve, dead-in-5
  plan.py         categorical macro-CEM planner + SAE-feature steering
  aq_sae.py       192-dim SAE via aquin + a real λ feature-diff
  tracker.py      stage board probed from real state
docs/             findings write-ups
tests/            161 checks; most pin a measurement, not an interface
```

## The dataset

**The 27.4 GB dataset is not in this repository** — GitHub is the wrong home for it (100 MB per
file, and it would be ~27 GB of binary blobs in history). It lives on a Modal volume and is
regenerated by the Stage 1 commands above, which are deterministic given the seeds in
`recipe.yaml`.

| | |
|---|---|
| 8,000 episodes | 4,000 jump-biased random + 4,000 from a trained PPO explorer |
| 4,877,368 observations | 224×224 RGB at frame-skip 2 |
| 27.4 GB compressed | 992 GB of raw pixels; NES artwork deflates 27× |
| per frame | buttons held, and `world_x`, `world_y`, `scroll`, `power`, `dies_in_5` from console RAM |

See [`data/README.md`](data/README.md) for the shard schema, the RAM validation protocol, and how to
mirror it to Hugging Face if you want a downloadable copy.

## Licence

Code: MIT (see [`LICENSE`](LICENSE)). The paper in [`paper/`](paper/) is redistributed under
CC BY-SA 4.0, © Balestriero & LeCun — see [`paper/README.md`](paper/README.md).

## Citation

```bibtex
@software{aqmario2026,
  title  = {AQ-Mario: SIGReg in an action-conditioned world model},
  year   = {2026},
  url    = {https://github.com/sachin1705s/aq-mario}
}
```
