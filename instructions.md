# AQ-Mario

A JEPA world model for Super Mario Bros 1-1, run as an Aquin
interpretability-in-the-loop demonstration. The claim: a model that predicts
perfectly can still fail to plan, and an interp gate catches that before you
waste a training run. Mario is the vehicle; the reusable gate harness is the
product.

This train fits `method: jepa` (methods/jepa.py). `eval.min_score: 1.0` is the
gate — it fails the run when any probe R² falls under its threshold, which is
how LeMario's y-probe collapse (R²=0.188) gets caught at epoch 2 instead of
after planning fails.

See STAGES.md for the four stages, STATUS.md for live progress.
