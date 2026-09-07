"""
Central config for AQ-Mario. Everything downstream (data schema, model dims,
eval gates, the stage tracker) reads from here so the stages stay consistent.

Design notes tied to the LeMario postmortem:
- FRAME_SKIP is 2, not 5. A jump arc is ~30 emulator frames; skip-5 gives ~6
  samples per jump and lets SIGReg throw away vertical info because it barely
  helps 1-step prediction. Skip-2 makes y matter for the prediction loss.
- We log RAM-derived state (x, y, scroll, alive) as AUX TARGETS. In the
  aux-loss variant these get small prediction heads; in the pure variant they
  are used only by the eval gates / probes, never in the training loss.
"""
from dataclasses import dataclass, field
from pathlib import Path


# ---- Action space -----------------------------------------------------------
# gym-super-mario-bros COMPLEX_MOVEMENT is 12 actions, but LeMario used the raw
# 6-button representation (Left, Right, Up, Down, A, B). We keep the 6-bit
# button vector as the ground-truth action, and ALSO expose a small macro-action
# set for the planner (categorical CEM over Gaussian is a bad fit for buttons).
BUTTONS = ["left", "right", "up", "down", "A", "B"]  # order is load-bearing; probes/action-encoder assume it
N_BUTTONS = len(BUTTONS)

MACRO_ACTIONS = {
    "wait":            [0, 0, 0, 0, 0, 0],
    "run_right":       [0, 1, 0, 0, 0, 1],   # right + B (run)
    "run_right_jump":  [0, 1, 0, 0, 1, 1],   # right + B + A
    "jump":            [0, 0, 0, 0, 1, 0],   # A only
    "run_left":        [1, 0, 0, 0, 0, 1],
}
MACRO_NAMES = list(MACRO_ACTIONS.keys())


# ---- Data schema ------------------------------------------------------------
@dataclass
class DataConfig:
    level: str = "SuperMarioBros-1-1-v0"   # stay on ONE level until it works
    frame_size: int = 224                   # matches LeMario / ViT-Tiny patch=14
    frame_skip: int = 2                     # emulator frames between observations
    obs_per_sample: int = 4                 # frames per training sample (context+targets)
    episodes_random: int = 4000             # random policy: jumps a lot, covers hazards
    episodes_ppo: int = 4000                # noisy PPO: covers "making progress"
    episodes_human: int = 0                 # optional, if you record any

    # WAS 256. A PPO episode at 224px/skip-2 is ~166 MB of frames; 256 of them
    # held in a list before writing is ~42 GB resident -> container OOM.
    # 16 keeps a shard near ~2.7 GB peak. See tracker's projected-size readout.
    shard_size: int = 16

    data_dir: Path = Path("~/.aqmario/data").expanduser()

    # Rough per-episode emulator-frame counts, used ONLY by the tracker to
    # project dataset size before you spend an hour generating it.
    est_frames_random: int = 400
    est_frames_ppo: int = 2200


# ---- Model ------------------------------------------------------------------
@dataclass
class ModelConfig:
    latent_dim: int = 192                   # the famous 192 numbers
    encoder: str = "vit_tiny_patch14"       # ~5M params
    predictor_layers: int = 6
    predictor_heads: int = 16
    predictor_dropout: float = 0.1          # paper ablation: 0.1 is the sweet spot
    history_len: int = 3                    # frames the predictor attends over
    action_frames: int = None               # set in __post_init__ from frame_skip
    pred_horizon: int = 3                   # predict next 3 latents

    def __post_init__(self):
        self.action_frames = DataConfig().frame_skip


# ---- Loss -------------------------------------------------------------------
@dataclass
class LossConfig:
    variant: str = "aux"                    # "pure" | "aux"
    sigreg_lambda: float = 0.1              # paper default; only effective knob
    sigreg_projections: int = 1024
    sigreg_knots: int = 17
    aux_y_lambda: float = 0.05              # Mario vertical position
    aux_scroll_lambda: float = 0.05         # camera scroll offset (fixes aliasing)
    aux_alive_lambda: float = 0.02          # dead-in-N signal

    # Stage 3 lambda-sweep. NOTE: this is SIGReg lambda, so each value is a FULL
    # training run (~3.2 h H100, ~$17), not a cheap SAE refit. Budget accordingly.
    sigreg_sweep: tuple = (0.01, 0.1, 0.5)


# ---- Eval gates -------------------------------------------------------------
@dataclass
class GateConfig:
    min_x_probe_r2: float = 0.95
    min_y_probe_r2: float = 0.80            # LeMario got 0.188 here — the target
    min_scroll_probe_r2: float = 0.90
    aliasing_px: int = 500
    min_aliasing_margin: float = 0.10
    probe_val_frac: float = 0.2
    probe_trajectories: int = 60            # Bai's split protocol, for comparability


# ---- SAE (Stage 3 inspect) --------------------------------------------------
@dataclass
class SAEConfig:
    d_model: int = 192                      # MUST be passed explicitly to train_sae;
                                            # the `aquin sae train` CLI cannot set it
    n_features: int = 2048                  # aquin default is 32768 -> 170x overcomplete at d=192
    placeholder_model_id: str = "gpt2"      # satisfies aquin's catalog config lookup only
    chunk_vectors: int = 65536              # rows per chunk_*.pt


# ---- Targets we are trying to beat (LeMario) --------------------------------
TARGETS = {
    "mse_1step":       dict(lemario=0.01377, target=0.012,  cmp="lt"),
    "mse_5step":       dict(lemario=0.07772, target=0.05,   cmp="lt"),
    "gain_5step":      dict(lemario=0.455,   target=0.60,   cmp="gt"),
    "x_probe_r2":      dict(lemario=0.997,   target=0.97,   cmp="gt"),
    "y_probe_r2":      dict(lemario=0.188,   target=0.90,   cmp="gt"),
    "scroll_probe_r2": dict(lemario=None,    target=0.90,   cmp="gt"),
}


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    seed: int = 0
    run_dir: Path = Path("~/.aqmario/runs").expanduser()


CFG = Config()
