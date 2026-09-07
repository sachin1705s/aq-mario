"""
Stage 2 adapter: stream training curves into `aquin watch`.

This is one of exactly TWO places aquin's tooling is model-agnostic enough to
use on a 192-dim ViT-JEPA (the other is aq_sae.py). `aquin watch` is a passive
external-metrics observer — it ingests a JSONL and knows nothing about your
architecture, so it fits without any faking.

    from aqmario.aq_watch import MetricsWriter
    mw = MetricsWriter(run_dir)                 # writes metrics.jsonl
    mw.log(step=i, epoch=e, pred_loss=..., sigreg_loss=..., eff_dim=...)

Then, once (needs `aquin login` first):
    aquin watch init --model gpt2                  -> run_id
    aquin watch ingest --run <id> --file metrics.jsonl --follow --step-field step
    aquin watch <run_id>                            # live tail

Two live failure signatures worth watching for:
  * sigreg_loss plateaus high  -> encoder can't reach Gaussian
  * pred_loss -> 0 while eff_dim collapses -> collapsing despite SIGReg, lambda too low
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def effective_dim(z) -> float:
    """
    Participation ratio of the latent covariance spectrum: (sum L)^2 / sum(L^2).
    192 means the latent is fully used; a collapse shows up here long before
    pred_loss looks wrong.
    """
    import torch
    z = z.detach().float()
    z = z - z.mean(0, keepdim=True)
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).clamp_min(0)
    s = ev.sum()
    return float(s * s / ev.pow(2).sum().clamp_min(1e-12))


class MetricsWriter:
    """Append-only JSONL, one object per step. Flushed so `--follow` sees it."""

    def __init__(self, run_dir: str | Path, name: str = "metrics.jsonl"):
        self.path = Path(run_dir).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self.path = self.path / name
        self._fh = self.path.open("a", buffering=1)

    def log(self, **row):
        self._fh.write(json.dumps(row, default=float) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def watch_init(model_id: str = "gpt2") -> str | None:
    """
    Register a watch run and return its id. `--model` is a catalog label only —
    aquin never loads it for a watch run, it just tags the stream.
    Requires `aquin login` and an active session.
    """
    r = subprocess.run(["aquin", "watch", "init", "--model", model_id],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[aq_watch] watch init failed: {r.stderr.strip() or r.stdout.strip()}")
        return None
    for tok in r.stdout.split():
        if len(tok) >= 8 and tok.strip().isascii() and "-" in tok:
            return tok.strip()
    print(r.stdout)
    return None


def ingest(run_id: str, metrics_path: str | Path, follow: bool = True,
           step_field: str = "step") -> subprocess.Popen:
    """Start `aquin watch ingest` as a background process tailing metrics.jsonl."""
    cmd = ["aquin", "watch", "ingest", "--run", run_id,
           "--file", str(metrics_path), "--step-field", step_field]
    if follow:
        cmd.append("--follow")
    return subprocess.Popen(cmd)
