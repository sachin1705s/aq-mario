"""
AQ-Mario stage tracker.

Four stages, each with checks that PROBE REAL STATE — files on disk, shard
counts, metrics.jsonl contents, gate result JSON. Nothing here is hand-ticked,
so the board cannot drift from reality.

    python -m aqmario.tracker            # print the board
    python -m aqmario.tracker --json     # machine-readable
    python -m aqmario.tracker --stage 2  # one stage
    python -m aqmario.tracker --write    # refresh STATUS.md

States:  DONE  PARTIAL  TODO  BLOCKED (a dependency is unmet)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from aqmario.config import CFG, TARGETS

ROOT = Path(__file__).resolve().parents[1]
RUNS = CFG.run_dir
DATA = CFG.data.data_dir

DONE, PARTIAL, TODO, BLOCKED = "DONE", "PARTIAL", "TODO", "BLOCKED"
FLAGPOLE_X = 3161            # world_x at the 1-1 flagpole


@dataclass
class Check:
    key: str
    label: str
    probe: Callable[[], tuple]      # -> (state, detail)
    weight: int = 1

    def run(self):
        try:
            return self.probe()
        except Exception as e:
            return TODO, f"probe error: {type(e).__name__}: {e}"


@dataclass
class Stage:
    num: int
    name: str
    goal: str
    exit_criteria: str
    checks: list = field(default_factory=list)

    def evaluate(self):
        rows = [(c, *c.run()) for c in self.checks]
        tot = sum(c.weight for c in self.checks)
        got = sum(c.weight for c, s, _ in rows if s == DONE)
        half = sum(c.weight for c, s, _ in rows if s == PARTIAL) * 0.5
        return rows, (got + half) / max(1, tot)


# ---------- small probe helpers ---------------------------------------------
def _src(name: str) -> Path:
    return ROOT / name


def _has_symbols(relpath: str, *symbols: str):
    """DONE if the file exists and defines every symbol; PARTIAL if some."""
    p = _src(relpath)
    if not p.is_file():
        return TODO, "not written"
    txt = p.read_text()
    if txt.count("\n") < 5:
        return TODO, "stub"
    found = [s for s in symbols if f"def {s}" in txt or f"class {s}" in txt]
    if not symbols:
        return DONE, f"{txt.count(chr(10))} lines"
    if len(found) == len(symbols):
        return DONE, f"{', '.join(found)}"
    if found:
        return PARTIAL, f"has {', '.join(found)}; missing {', '.join(set(symbols) - set(found))}"
    return PARTIAL, f"file present, missing {', '.join(symbols)}"


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _latest_metrics():
    cands = sorted(RUNS.glob("*/metrics.jsonl")) if RUNS.exists() else []
    if not cands:
        return None, []
    p = max(cands, key=lambda q: q.stat().st_mtime)
    rows = []
    for line in p.read_text().splitlines():
        r = _read_json_line(line)
        if r:
            rows.append(r)
    return p, rows


def _read_json_line(line):
    try:
        return json.loads(line)
    except Exception:
        return None


def _gate_results():
    """Newest gate result json written by scripts/run_gates.py."""
    cands = sorted(RUNS.glob("*/gates_*.json")) if RUNS.exists() else []
    if not cands:
        return None
    return _read_json(max(cands, key=lambda q: q.stat().st_mtime))


# =============================================================================
# STAGE 1 — Ground truth
# =============================================================================
def p_ram_module():
    return _has_symbols("aqmario/ram.py", "ram_state", "validated")


def p_ram_validated():
    from aqmario import ram
    lock = ram.load_lock()
    if ram.validated():
        chosen = {**ram.CHOSEN, **(lock.get("chosen", {}) if lock else {})}
        return DONE, f"locked: world_y={chosen['world_y']}"
    unl = [f for f, v in ram.CHOSEN.items() if not v]
    return TODO, f"unvalidated: {', '.join(unl)} — run scripts/validate_ram.py --lock"


def p_collector():
    return _has_symbols("scripts/collect_data.py", "collect", "save_shard")


def p_shards():
    if not DATA.exists():
        return TODO, "no data dir yet"
    shards = list(DATA.rglob("shard_*.npz"))
    if not shards:
        return TODO, "0 shards"
    gb = sum(s.stat().st_size for s in shards) / 1e9
    eps = len(shards) * CFG.data.shard_size
    target = CFG.data.episodes_random + CFG.data.episodes_ppo
    pct = eps / target
    state = DONE if pct >= 1.0 else PARTIAL
    return state, f"{len(shards)} shards ~{eps}/{target} eps, {gb:.1f} GB on disk"


def p_data_budget():
    d = CFG.data
    obs = (d.episodes_random * d.est_frames_random + d.episodes_ppo * d.est_frames_ppo) / d.frame_skip
    raw = obs * d.frame_size ** 2 * 3 / 1e9
    peak = d.shard_size * d.est_frames_ppo / d.frame_skip * d.frame_size ** 2 * 3 / 1e9
    if peak > 8:
        return TODO, f"shard_size={d.shard_size} -> {peak:.0f} GB peak RAM, will OOM"
    if raw > 500:
        return PARTIAL, f"{obs/1e6:.1f}M obs / {raw:.0f} GB raw — I/O bound; consider frame_size 112 or fewer eps"
    return DONE, f"{obs/1e6:.1f}M obs, {raw:.0f} GB raw, {peak:.1f} GB peak shard RAM"


def p_modal_data():
    return _has_symbols("modal_app.py", "collect_shard")


# =============================================================================
# STAGE 2 — World model
# =============================================================================
def p_model():
    return _has_symbols("aqmario/model.py", "Encoder", "ActionEncoder", "Predictor")


def p_losses():
    return _has_symbols("aqmario/losses.py", "sigreg", "aux_heads")


def p_param_count():
    p = RUNS / "param_count.json"
    d = _read_json(p) if p.is_file() else None
    if not d:
        return TODO, "no dry run yet"
    n = d.get("total", 0) / 1e6
    ok = 10 <= n <= 22
    return (DONE if ok else PARTIAL), f"{n:.1f}M params (target ~15M)"


def p_dryrun():
    d = _read_json(RUNS / "dryrun.json") if (RUNS / "dryrun.json").is_file() else None
    if not d:
        return TODO, "single-shard dry run not done"
    moved = d.get("pred_loss_moved") and d.get("sigreg_loss_moved")
    return (DONE if moved else PARTIAL), f"pred moved={d.get('pred_loss_moved')} sigreg moved={d.get('sigreg_loss_moved')}"


def p_metrics():
    p, rows = _latest_metrics()
    if not rows:
        return TODO, "no metrics.jsonl"
    last = rows[-1]
    keys = [k for k in ("pred_loss", "sigreg_loss", "eff_dim") if k in last]
    detail = " ".join(f"{k}={last[k]:.4g}" for k in keys if isinstance(last.get(k), (int, float)))
    return (DONE if len(keys) >= 2 else PARTIAL), f"{len(rows)} steps · {detail}"


def p_watch():
    wr = Path("~/.aquin/watch").expanduser()
    if not wr.exists():
        return TODO, "no aquin watch runs (needs `aquin login` + `session start`)"
    runs = [d for d in wr.iterdir() if d.is_dir()]
    if not runs:
        return TODO, "watch dir empty"
    ev = sum(len((d / "events.jsonl").read_text().splitlines())
             for d in runs if (d / "events.jsonl").is_file())
    return (DONE if ev else PARTIAL), f"{len(runs)} run(s), {ev} events ingested"


def p_epochs():
    _, rows = _latest_metrics()
    if not rows:
        return TODO, "not started"
    ep = max((r.get("epoch", 0) for r in rows), default=0)
    return (DONE if ep >= 10 else PARTIAL if ep else TODO), f"epoch {ep}/12"


def p_aq_method():
    """recipe.yaml `method: jepa` resolves to methods/jepa.py in THIS train
    (protocol/method.py checks train/methods/ before kernel/methods/)."""
    m = ROOT / "methods" / "jepa.py"
    r = ROOT / "recipe.yaml"
    if not m.is_file():
        return TODO, "methods/jepa.py not written"
    if not r.is_file() or "method: jepa" not in r.read_text():
        return PARTIAL, "methods/jepa.py exists but recipe.yaml does not select it"
    txt = m.read_text()
    need = ["fit", "evaluate", "predict", "write_inspect"]
    miss = [n for n in need if f"def {n}" not in txt]
    return (DONE, "fit/evaluate/predict/write_inspect") if not miss else (PARTIAL, f"missing {', '.join(miss)}")


def p_adapters():
    ok = []
    for f, sym in (("aqmario/aq_watch.py", "MetricsWriter"), ("aqmario/aq_sae.py", "train_latent_sae")):
        pth = ROOT / f
        if pth.is_file() and sym in pth.read_text():
            ok.append(Path(f).stem)
    if len(ok) == 2:
        return DONE, "aq_watch + aq_sae"
    return (PARTIAL, f"have {', '.join(ok)}") if ok else (TODO, "no aq adapters")


# =============================================================================
# STAGE 3 — Gates + inspect
# =============================================================================
def p_gates_code():
    return _has_symbols("aqmario/gates.py", "probe_gate", "aliasing_gate", "dead_in_5_gate")


def p_gate_runner():
    return _has_symbols("scripts/run_gates.py", "main")


def p_gate_numbers():
    g = _gate_results()
    if not g:
        return TODO, "no gate run yet"
    bits, allok = [], True
    for k, gate_min in (("x_probe_r2", CFG.gate.min_x_probe_r2),
                        ("y_probe_r2", CFG.gate.min_y_probe_r2),
                        ("scroll_probe_r2", CFG.gate.min_scroll_probe_r2)):
        v = g.get(k)
        if v is None:
            allok = False
            bits.append(f"{k}=?")
            continue
        ok = v >= gate_min
        allok &= ok
        bits.append(f"{k}={v:.3f}{'' if ok else f'<{gate_min}'}")
    return (DONE if allok else PARTIAL), " ".join(bits)


def p_beats_lemario():
    g = _gate_results()
    if not g:
        return TODO, "no numbers yet"
    beaten, total = 0, 0
    for k, t in TARGETS.items():
        v = g.get(k)
        if v is None:
            continue
        total += 1
        beaten += (v < t["target"]) if t["cmp"] == "lt" else (v > t["target"])
    if not total:
        return TODO, "no comparable metrics"
    y = g.get("y_probe_r2")
    head = f"y-probe {y:.3f} vs LeMario 0.188" if y is not None else ""
    return (DONE if beaten == total else PARTIAL), f"{beaten}/{total} targets met · {head}"


def p_aq_eval():
    """The gate is native: recipe eval.min_score + methods/jepa.py:evaluate()."""
    r = ROOT / "recipe.yaml"
    if not r.is_file():
        return TODO, "no recipe.yaml — run `aq init .`"
    txt = r.read_text()
    has_min = "min_score" in txt and "min_score: null" not in txt
    has_eval = (ROOT / "methods" / "jepa.py").is_file() and "def evaluate" in (ROOT / "methods" / "jepa.py").read_text()
    if has_min and has_eval:
        return DONE, "recipe eval.min_score + methods/jepa.py:evaluate() — fails closed"
    bits = []
    if not has_min:
        bits.append("recipe eval.min_score unset")
    if not has_eval:
        bits.append("methods/jepa.py:evaluate() missing")
    return PARTIAL, "; ".join(bits)


def p_latents():
    dirs = sorted(RUNS.glob("lat/*")) if RUNS.exists() else []
    if not dirs:
        return TODO, "no latents dumped"
    tot = 0
    for d in dirs:
        tot += len(list(d.glob("chunk_*.pt")))
    return (DONE if tot else PARTIAL), f"{len(dirs)} dir(s), {tot} chunk_*.pt"


def p_saes():
    saes = sorted(RUNS.glob("sae/*.pt")) if RUNS.exists() else []
    want = len(CFG.loss.sigreg_sweep)
    if not saes:
        return TODO, f"0/{want} SAEs (one per SIGReg lambda)"
    return (DONE if len(saes) >= want else PARTIAL), f"{len(saes)}/{want} SAEs trained"


def p_lambda_diff():
    p = RUNS / "sae" / "lambda_diff.json"
    d = _read_json(p) if p.is_file() else None
    if not d:
        return TODO, "no lambda-sweep feature diff"
    return DONE, f"{len(d.get('features', []))} features compared across {CFG.loss.sigreg_sweep}"


# =============================================================================
# STAGE 4 — Control
# =============================================================================
def p_planner():
    return _has_symbols("aqmario/plan.py", "macro_cem", "probe_cost")


def p_steer():
    d = _read_json(RUNS / "steer_demo.json") if (RUNS / "steer_demo.json").is_file() else None
    if not d:
        return TODO, "y-feature steering not demonstrated"
    return (DONE if d.get("rollout_jumped") else PARTIAL), f"delta_y={d.get('delta_y')}"


def p_nearby_goal():
    d = _read_json(RUNS / "plan_sanity.json") if (RUNS / "plan_sanity.json").is_file() else None
    if not d:
        return TODO, "nearby-goal sanity plan not run"
    return (DONE if d.get("reached") else PARTIAL), f"reached={d.get('reached')} in {d.get('steps')} steps"


def p_clear_11():
    best = _read_json(RUNS / "plan_best.json") if (RUNS / "plan_best.json").is_file() else None
    if not best:
        return TODO, f"0/{FLAGPOLE_X} world_x"
    x = best.get("max_world_x", 0)
    return (DONE if best.get("cleared") else PARTIAL), f"max world_x {x}/{FLAGPOLE_X} ({x/FLAGPOLE_X:.0%})"


def p_gif():
    g = list(ROOT.glob("demo*.gif")) + list(RUNS.glob("*.gif")) if RUNS.exists() else list(ROOT.glob("demo*.gif"))
    return (DONE, f"{g[0].name}") if g else (TODO, "no demo GIF")


# =============================================================================
STAGES = [
    Stage(1, "Ground truth",
          "RAM extraction you can trust, and 8k episodes of 1-1 behind it.",
          "RAM addresses locked by a real trace AND >=8k episodes sharded.",
          [Check("ram.py", "ram.py — single RAM->state definition", p_ram_module),
           Check("ram.valid", "RAM addresses validated on a 60-frame trace", p_ram_validated, 2),
           Check("collector", "collect_data.py emulator + logging", p_collector),
           Check("budget", "dataset size / shard RAM projection", p_data_budget),
           Check("modal", "modal_app.py CPU fan-out for data-gen", p_modal_data),
           Check("shards", "episodes collected", p_shards, 2)]),

    Stage(2, "World model",
          "A ~15M JEPA that predicts 1-1 dynamics in 192-dim latent space.",
          "10+ epochs trained bf16, both losses moving, curves live in aquin watch.",
          [Check("model", "model.py encoder + action-enc + AdaLN-Zero predictor", p_model),
           Check("losses", "losses.py SIGReg + aux heads", p_losses),
           Check("params", "parameter count in ~15M band", p_param_count),
           Check("dryrun", "single-shard dry run, both losses move", p_dryrun),
           Check("metrics", "metrics.jsonl streaming", p_metrics),
           Check("watch", "aquin watch ingesting curves", p_watch),
           Check("aqmethod", "methods/jepa.py wired as recipe `method: jepa`", p_aq_method),
           Check("adapters", "aq_watch / aq_sae adapters", p_adapters),
           Check("epochs", "full run 10+ epochs", p_epochs, 2)]),

    Stage(3, "Gates + inspect",
          "The Aquin claim: catch a planning failure from the representation alone.",
          "Gates run per checkpoint, y-probe R2 > 0.80, SAE lambda-diff table produced.",
          [Check("gates", "gates.py probe / aliasing / dead-in-5", p_gates_code),
           Check("runner", "run_gates.py standalone on a checkpoint", p_gate_runner),
           Check("numbers", "gate thresholds met", p_gate_numbers, 2),
           Check("lemario", "beats LeMario's published numbers", p_beats_lemario, 2),
           Check("aqeval", "gate fails closed via recipe eval.min_score", p_aq_eval, 2),
           Check("latents", "latents dumped as chunk_*.pt", p_latents),
           Check("sae", "SAE trained per SIGReg lambda", p_saes),
           Check("lamdiff", "lambda-sweep feature diff table", p_lambda_diff, 2)]),

    Stage(4, "Control",
          "Plan with the model, and steer it with a feature the SAE found.",
          "Sub-goal chain clears World 1-1; y-feature steering changes the rollout.",
          [Check("plan", "plan.py macro-CEM + probe cost", p_planner),
           Check("steer", "y-feature steering changes rollout", p_steer),
           Check("sanity", "nearby-goal sanity plan", p_nearby_goal),
           Check("clear", "clears World 1-1", p_clear_11, 3),
           Check("gif", "demo GIF", p_gif)]),
]

BAR = {DONE: "#", PARTIAL: "+", TODO: ".", BLOCKED: "x"}
MARK = {DONE: "[x]", PARTIAL: "[~]", TODO: "[ ]", BLOCKED: "[!]"}


def render(stages, use_color=True) -> str:
    C = (lambda s, c: f"\033[{c}m{s}\033[0m") if use_color else (lambda s, c: s)
    col = {DONE: "32", PARTIAL: "33", TODO: "90", BLOCKED: "31"}
    out = []
    prev_done = True
    overall = []
    for st in stages:
        rows, pct = st.evaluate()
        overall.append(pct)
        filled = int(round(pct * 24))
        bar = "#" * filled + "." * (24 - filled)
        head = f"STAGE {st.num} — {st.name}"
        out.append("")
        out.append(C(head, "1") + "  " + C(f"[{bar}] {pct:6.0%}", col[DONE if pct == 1 else PARTIAL if pct else TODO]))
        out.append(f"  goal  {st.goal}")
        out.append(f"  exit  {st.exit_criteria}")
        if not prev_done:
            out.append(C("  (upstream stage incomplete — numbers here are provisional)", "90"))
        out.append("")
        for c, state, detail in rows:
            out.append(f"    {C(MARK[state], col[state])} {c.label:<48} {C(detail, '90')}")
        prev_done = pct >= 1.0
    tot = sum(overall) / len(overall)
    out.append("")
    out.append(C(f"OVERALL  [{'#' * int(round(tot*24)) + '.' * (24 - int(round(tot*24)))}] {tot:.0%}", "1"))
    return "\n".join(out)


def as_dict(stages):
    d = {"stages": [], "overall": 0.0}
    for st in stages:
        rows, pct = st.evaluate()
        d["stages"].append({
            "num": st.num, "name": st.name, "goal": st.goal,
            "exit_criteria": st.exit_criteria, "progress": round(pct, 3),
            "checks": [{"key": c.key, "label": c.label, "state": s, "detail": det}
                       for c, s, det in rows],
        })
    d["overall"] = round(sum(s["progress"] for s in d["stages"]) / len(d["stages"]), 3)
    return d


def write_status_md(stages, path: Path):
    d = as_dict(stages)
    L = [f"# AQ-Mario — status\n", f"**Overall: {d['overall']:.0%}**\n"]
    L.append("| Stage | Progress | Exit criteria |")
    L.append("|---|---|---|")
    for s in d["stages"]:
        L.append(f"| {s['num']}. {s['name']} | {s['progress']:.0%} | {s['exit_criteria']} |")
    for s in d["stages"]:
        L.append(f"\n## Stage {s['num']} — {s['name']} · {s['progress']:.0%}\n")
        L.append(f"_{s['goal']}_\n")
        for c in s["checks"]:
            L.append(f"- {MARK[c['state']]} **{c['label']}** — {c['detail']}")
    L.append("\n---\n_Generated by `python -m aqmario.tracker --write`. "
             "Every line is probed from real state, never hand-ticked._")
    path.write_text("\n".join(L) + "\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aqmario.tracker")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write", action="store_true", help="refresh STATUS.md")
    ap.add_argument("--stage", type=int, choices=[1, 2, 3, 4])
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args(argv)
    stages = [s for s in STAGES if not a.stage or s.num == a.stage]
    if a.json:
        print(json.dumps(as_dict(stages), indent=2))
        return 0
    if a.write:
        p = write_status_md(STAGES, ROOT / "STATUS.md")
        print(f"wrote {p}")
    print(render(stages, use_color=not a.no_color and sys.stdout.isatty()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
