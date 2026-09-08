# The paper this project implements

**LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics**
Randall Balestriero, Yann LeCun — arXiv:2511.08544, 11 November 2025

- Paper: <https://arxiv.org/abs/2511.08544>
- Authors' implementation: <https://github.com/rbalestr-lab/lejepa>
- Local copy: [`LeJEPA-2511.08544.pdf`](LeJEPA-2511.08544.pdf)

## Licence of the included PDF

The paper is distributed under **CC BY-SA 4.0**
(<https://creativecommons.org/licenses/by-sa/4.0/>). It is redistributed here
unmodified, with attribution to Balestriero and LeCun, under the terms of that
licence. The code in this repository is an independent implementation inspired by
the paper, not a derivative of the paper's text, and carries its own licence.

## What this repository takes from it

LeJEPA identifies the **isotropic Gaussian** as the optimal embedding
distribution for minimising downstream prediction risk, and introduces
**SIGReg** (Sketched Isotropic Gaussian Regularization) to reach it: project
embeddings onto random directions and match each 1-D marginal to a standard
normal via a characteristic-function test. The claim is that this removes the
need for the anti-collapse heuristics — stop-gradients, EMA teachers,
asymmetric predictors — while keeping linear complexity and bounded gradients.

This repository applies SIGReg to a domain the paper does not evaluate: an
**action-conditioned world model over video**, judged not by linear-probe
accuracy on classes but by whether a *control-relevant state variable* survives
in the representation.

## What we measured, and where it agrees or disagrees

| the claim, as we understand it | what we measured | see |
|---|---|---|
| SIGReg makes stop-gradients / EMA unnecessary | **Did not hold at our scale.** Without stop-grad the latent went to rank 2 while reporting a 5-step gain of 0.97 | [`../docs/lejepa-findings.md`](../docs/lejepa-findings.md) §4 |
| λ is the single trade-off knob | λ alone could not prevent collapse; the incentive to collapse outweighed it 4–45× | §4 |
| SIGReg prevents collapse | It does — but its *power to detect* collapse is ≈linear in batch size (2.8× null at n=32 vs 41.8× at n=512) | §2 |
| — | The statistic has a **closed-form null**, `0.51914/n`, without which its value and λ are not comparable across batch sizes | §1 |
| — | A trailing BatchNorm (which the architecture needs) cuts the collapse signal 254× → 10× | §3 |
| — | On temporally correlated video batches the iid assumption breaks; a *healthy* latent reads 6.7× null | §5 |
| — | Preventing collapse did **not** preserve usable state: y-probe R² = −0.383 on an uncollapsed representation | §6 |

> **Caveat, stated plainly.** Our characterisation of the paper's claims in the
> left column is a paraphrase from reading it, not a set of quotations. The
> measurements in the right column are ours and are reproducible from this
> repository. Before citing any disagreement, check the left column against the
> paper text. Our results also come from ~4% of a full training schedule on a
> single environment, so "did not hold at our scale" is a statement about our
> scale, not a refutation.

## Citation

```bibtex
@article{balestriero2025lejepa,
  title   = {LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics},
  author  = {Balestriero, Randall and LeCun, Yann},
  journal = {arXiv preprint arXiv:2511.08544},
  year    = {2025},
  url     = {https://arxiv.org/abs/2511.08544}
}
```
