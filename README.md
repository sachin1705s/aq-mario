# AQ-Mario

A JEPA world model for Super Mario Bros 1-1, built as an Aquin
interpretability-in-the-loop demonstration. The claim is not "playable Mario" —
it is that **a model which predicts perfectly can still fail to plan, and an
interp gate catches that before you waste a training run.**

The thing we're beating is Benjamin Bai's [LeMario](https://www.benjamin-bai.com/projects/lemario),
whose encoder scored **y-probe R² = 0.188** — it barely knew Mario's height, so
any goal needing a jump was invisible to the planner.

| Metric | LeMario | Target |
|---|---|---|
| 1-step latent MSE | 0.01377 | < 0.012 |
| 5-step latent MSE | 0.07772 | < 0.05 |
| x-probe R² | 0.997 | ≥ 0.97 (hold) |
| **y-probe R²** | **0.188** | **> 0.90** |
| scroll-probe R² | unmeasured | > 0.90 |
| Clear 1-1 by planning | never | first time |

## Four stages

See **[STAGES.md](STAGES.md)** for the full definition of each, and
**[STATUS.md](STATUS.md)** for where things actually stand.

1. **Ground truth** — validated RAM extraction, 8k episodes of 1-1
2. **World model** — ~15M JEPA, 192-dim latent, SIGReg + aux heads
3. **Gates + inspect** — probe / aliasing / dead-in-5 gates, SAE λ-diff
4. **Control** — macro-CEM planner + feature steering → clear 1-1

## Tracker

Progress is derived from real state, never hand-ticked.

```bash
python -m aqmario.tracker             # the board
python -m aqmario.tracker --stage 3   # one stage
python -m aqmario.tracker --write     # refresh STATUS.md
```

## Layout

```
recipe.yaml          aq train spec — method: jepa, eval.min_score is the gate
methods/jepa.py      aq method adapter (train dir wins over kernel/methods/)
aqmario/
  config.py          dims, loss weights, gate thresholds, LeMario targets
  ram.py             ONE RAM->state definition; fails closed until validated
  tracker.py         the four-stage tracker
  aq_watch.py        metrics.jsonl + `aquin watch` ingest + eff_dim
  aq_sae.py          latents -> chunk_*.pt -> train_sae(d_model=192) -> λ-diff
  model.py           [stage 2] encoder + action-enc + AdaLN-Zero predictor
  losses.py          [stage 2] SIGReg + aux heads
  gates.py           [stage 3] probe / aliasing / dead-in-5
  plan.py            [stage 4] macro-CEM + probe cost + steering
scripts/
  validate_ram.py    DO THIS FIRST — locks the RAM addresses
  collect_data.py    emulator + RAM logging -> shards
  train.py           [stage 2]
  run_gates.py       [stage 3]
modal_app.py         one entrypoint per stage
```

## Cost

~$125 realistic end-to-end on Modal (data-gen $0.62, ~$17 per H100 training
run, gates/SAE/planning under $6). Storage is free — 196 GB compressed sits
under the 1 TiB volume tier. Wall clock ~8–10 h of compute; the λ-sweep runs in
parallel so it costs money but no extra time.

## Run order

```bash
python scripts/validate_ram.py --lock          # blocks everything downstream
modal run modal_app.py::stage1_data
modal run modal_app.py::stage2_train --variant aux --epochs 12
modal run modal_app.py::stage3_gates --ckpt /runs/aux/ep12.pt
modal run modal_app.py::stage3_sweep           # 3 full runs, one per SIGReg λ
modal run modal_app.py::stage4_plan --ckpt ...
```
