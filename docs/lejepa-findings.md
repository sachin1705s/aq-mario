# LeJEPA in a world model: what transfers, what doesn't, and the missing calibration

This project is an **application of LeJEPA** (Balestriero & LeCun) to a domain the
paper does not test: a world model for control, trained on video with actions,
and evaluated by whether a planner-relevant state variable survives in the
representation. LeMario is a useful reference point but it is not the subject —
the subject is SIGReg, and what happens to it outside image SSL.

Everything below is measured on Super Mario Bros 1-1: 8,000 episodes, 4.88M
observations, a 9.8M-parameter ViT-Tiny JEPA with a 192-dim latent.

> Status of the claims-as-stated: the LeJEPA claims below are as commonly
> summarised (SIGReg removes the stop-gradient/EMA heuristics; one trade-off
> hyperparameter; provable collapse prevention). Before publishing, each should
> be quoted from the paper text directly rather than from this summary — the
> measurements are ours, the paraphrases are not yet checked line by line.

---

## 1. The calibration the objective needs and does not ship with

**SIGReg's value is uninterpretable without its null.** Under the null
(z ~ N(0, I)), `E|φ_n(t) − φ(t)|² = (1 − e^{−t²})/n` exactly, so for the
Epps–Pulley statistic integrated against `e^{−t²}`:

```
E[T | z ~ N(0,I)] = √π · (1 − 1/√2) / n = 0.51914 / n
```

Verified against measurement within 3% from n = 64 to n = 2048. Consequences:

* At batch 64 the term **floors at 0.0013** however Gaussian the encoder gets.
  Driving it to zero is not a goal; reaching ~1.0× null is.
* Raw SIGReg values from two runs at different batch sizes are **not
  comparable**, so neither are their λ. `sigreg_ratio = T / null(n)` is.
* Because the floor comes from sampling noise, the optimiser can push T slightly
  *below* 1.0× by making the batch mildly repulsive. That is benign, and a second
  reason the target is "ratio near 1" rather than "loss near 0".

## 2. Its power to see a collapse is a function of batch size

Ratio to null for a rank-r latent after BN, D = 192:

| rank | n=32 | n=64 | n=128 | n=512 | n=2048 |
|---|---|---|---|---|---|
| 2 | 2.8× | 5.6× | 10.9× | 41.8× | 173.8× |
| 4 | 1.8× | 3.3× | 6.4× | 25.0× | 97.8× |
| 16 | 0.8× | 1.2× | 2.0× | 7.2× | 28.5× |
| 192 (healthy) | 0.4× | 0.5× | 0.5× | 1.0× | 2.8× |

At n=32 a fully collapsed latent reads 2.8× against a healthy 0.4× — barely
separable, so the term cannot push back on something it can hardly see. This is
the mechanism behind §4, and it puts SIGReg in direct tension with memory:
a step encodes `batch × window` images, so the batch a world model can afford is
exactly the regime where SIGReg is weakest. Activation checkpointing is not an
optimisation here, it is what buys the regulariser its power.

## 3. A trailing BatchNorm silently halves the job

Our encoder ends in `BatchNorm(affine=False)`, which is standard practice and is
also required here (a trailing LayerNorm puts z on a fixed-radius sphere and
fights the Gaussian target directly). But BN pins the covariance **diagonal** for
free, which is a large part of what SIGReg was detecting:

* rank-8 latent, raw: **254× null**
* the same latent after BN: **10× null**

Still unambiguous, but 25× less signal. BN and SIGReg are complementary rather
than reinforcing — BN fixes the diagonal, SIGReg fixes the rest — and a paper
reporting SIGReg values from a BN-projected encoder is reporting a different
quantity than one without.

The corresponding positive result, and the single experiment that says the design
is sound: SIGReg escapes a rank-8 collapse **only when the gradient flows through
the BN**, as it does in a real encoder. Optimising the latent as a free tensor
barely moves it.

| step | sigreg | ×null | eff_dim |
|---|---|---|---|
| 0 | 0.0139 | 10.0 | 7.9 |
| 100 | 0.0051 | 3.8 | 21.5 |
| 300 | 0.0021 | 1.6 | 52.0 |
| 600 | 0.0015 | 1.1 | 69.5 |

## 4. In this domain the stop-gradient was still required

The headline simplification — SIGReg replaces the stop-gradient and EMA-teacher
heuristics — did not hold at our scale. 500 steps, identical everything else:

| batch | stop-grad | eff_dim (init → 500) | gain_5step |
|---|---|---|---|
| 32 | ✗ | 7.9 → **2.0** | 0.969 *(fake)* |
| 128 | ✗ | 11.1 → **2.0** | 0.978 *(fake)* |
| 32 | ✓ | 7.9 → 5.8 | 0.624 |
| 128 | ✓ | 11.1 → **8.4** | 0.651 |

λ alone could not substitute for it. The arithmetic explains why: collapsing buys
the optimiser **0.80** of prediction loss, while λ=5's SIGReg term costs **0.18**
and the auxiliary terms **0.008**. It is outgunned 4–45×, depending on λ.

Why a world model is harder than image SSL here: with a shared encoder,
`MSE(ẑ, z_target)` is *directly minimised* by making `z_target` constant, because
the target is produced by the same trainable encoder. Collapse is not a failure
mode to be avoided, it is the objective's global optimum unless something removes
the incentive. Image SSL's positive-pair objectives have the same structure, but
the prediction task here is far easier to trivialise — one scalar suffices.

## 5. Temporally correlated batches break the null

A world-model batch is **B windows × W near-identical frames** (2 emulator frames
apart at skip-2). SIGReg's null assumes iid samples, so the effective count is
**B**, not B×W. Measured on a synthetic FULL-RANK latent batched as 32×6:

| scoring | reads | verdict |
|---|---|---|
| flattened, vs the n=192 null | 6.7× | "collapsing" |
| flattened, vs the honest n=32 null | 1.2× | fine |
| per-frame-position mean, n=32 null | 1.1× | fine |

Flattening also makes SIGReg penalise the within-window similarity the predictor
depends on — it pushes consecutive frames apart while the prediction loss pulls
them together. Computing the statistic per frame position and averaging keeps
every latent, makes the null exact, and removes the conflict. Anyone applying
LeJEPA to video should do this; it is not visible from the image-domain framing.

The same correction applies to the participation-ratio diagnostic: Marchenko–
Pastur gives `E[PR] = nD/(n+D)`, so a healthy 192-dim latent reads **128, not
192**, at n=384. Inverting it (`D_eff = PR·n/(n−PR)`) recovers 192.0 ± 1 at every
n and stays at 7.6 on a rank-8 latent.

## 6. The application-level finding: no collapse ≠ usable state

This is the part that matters for control, and it is not addressed by an
anti-collapse guarantee.

Our `pure` variant — SIGReg only, no auxiliary supervision — **did not collapse**
and still scored **y-probe R² = −0.383** on Mario's height, the single variable a
platformer planner needs most. (Full 8k dataset, 60 held-out trajectories, at 4%
of the training schedule; the full-budget control is running.)

| probe | pure (SIGReg only) | + auxiliary heads | LeMario |
|---|---|---|---|
| **height (y)** | **−0.383** | 0.796 | 0.188 |
| position (x) | 0.451 | 0.870 | 0.997 |
| camera (scroll) | 0.464 | 0.878 | — |

An isotropic-Gaussian embedding is a **floor**, not a sufficient condition: it
guarantees the representation is not degenerate, not that it contains what the
downstream task requires. A 0.07M-parameter auxiliary head recovers it.

Which is the argument for the thing this project is actually built around:
**gate on the representation, per checkpoint, not on the loss.** Every collapsed
and every y-blind model here reported an excellent loss. The pure run's failure
was visible from a probe at 4% of the training budget, before any planner
existed — which is the difference between a 35-minute answer and LeMario's
months-long one.

---

## What this suggests for practice

1. Report `sigreg / null(n)`, never raw SIGReg. Publish the batch size next to λ.
2. In video/RL, compute SIGReg per frame position, not over the flattened batch.
3. Treat batch size as a SIGReg hyperparameter, not just a throughput knob.
4. Do not assume the stop-gradient is removable at practical batch sizes outside
   image SSL — measure it, it is one ablation.
5. Anti-collapse is not representation quality. Gate on a probe of the state the
   downstream task needs.
