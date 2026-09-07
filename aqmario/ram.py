"""
Single source of truth for RAM -> interpretable state.

Imported by BOTH scripts/collect_data.py (logging) and aqmario/gates.py (probe
targets). If training targets and eval probes ever drift apart, every number in
the writeup is meaningless — so there is exactly one definition, here.

WHY CANDIDATES INSTEAD OF CONSTANTS
-----------------------------------
The original handoff's addresses were written from memory of the SMB RAM map.
A wrong address does not crash; it silently corrupts every gate, probe and
planner cost. So each field declares the candidate addresses that are plausible
in the datacrystal map, together with an INVARIANT that a real 60-frame
"walk right" trace must satisfy. scripts/validate_ram.py runs that trace and
reports which candidates hold. You then lock the winner into CHOSEN below.

Until validate_ram.py has been run and CHOSEN confirmed, `validated()` is False
and the Stage 1 tracker will refuse to mark data collection as trustworthy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

STATE_FIELDS = ("world_x", "world_y", "y_screen", "scroll", "alive", "power")

# ---- candidate addresses ----------------------------------------------------
# Each entry: name -> (page_addr, fine_addr) or (addr,) for single-byte reads.
CANDIDATES = {
    "world_x": {
        # standard, and what gym-super-mario-bros itself uses for x_position
        "page_6D_fine_86": (0x006D, 0x0086),
    },
    "world_y": {
        # 0x00CE = player y on screen; 0x00B5 = player vertical page ("HighPos")
        "page_B5_fine_CE": (0x00B5, 0x00CE),
        # 0x03B8 is the other commonly cited player-y-on-screen byte
        "page_B5_fine_03B8": (0x00B5, 0x03B8),
    },
    "scroll": {
        # 0x071A = screen-edge page loc, 0x071C = screen-edge x pos
        "page_071A_fine_071C": (0x071A, 0x071C),
    },
}

# Locked-in choice. Set by scripts/validate_ram.py --lock after the trace passes.
CHOSEN = {
    "world_x": "page_6D_fine_86",
    "world_y": None,      # <- UNVALIDATED. run scripts/validate_ram.py
    "scroll":  "page_071A_fine_071C",
}

PLAYER_STATE = 0x000E          # 0x06, 0x0B = dying / dead
POWER_STATE  = 0x0756          # 0=small, 1=big, 2=fire
DEAD_STATES  = (0x06, 0x0B)

_LOCK_PATH = Path("~/.aqmario/ram_lock.json").expanduser()


def _pair(ram, page_addr: int, fine_addr: int) -> int:
    return int(ram[page_addr]) * 256 + int(ram[fine_addr])


def load_lock() -> dict | None:
    """Return the validated address choice written by validate_ram.py --lock."""
    if _LOCK_PATH.is_file():
        try:
            return json.loads(_LOCK_PATH.read_text())
        except Exception:
            return None
    return None


def validated() -> bool:
    """True once every field has a confirmed candidate. Gates Stage 1."""
    lock = load_lock()
    chosen = {**CHOSEN, **(lock.get("chosen", {}) if lock else {})}
    return all(chosen.get(f) for f in ("world_x", "world_y", "scroll"))


def _resolve() -> dict:
    lock = load_lock()
    chosen = dict(CHOSEN)
    if lock:
        chosen.update(lock.get("chosen", {}))
    missing = [f for f, v in chosen.items() if not v]
    if missing:
        raise RuntimeError(
            f"RAM addresses not validated for {missing}. "
            f"Run: python scripts/validate_ram.py --lock"
        )
    return chosen


def ram_state(ram, chosen: dict | None = None) -> dict:
    """Extract interpretable ground-truth state from the NES RAM array."""
    c = chosen or _resolve()
    world_x = _pair(ram, *CANDIDATES["world_x"][c["world_x"]])
    y_page, y_fine = CANDIDATES["world_y"][c["world_y"]]
    world_y = _pair(ram, y_page, y_fine)
    scroll = _pair(ram, *CANDIDATES["scroll"][c["scroll"]])
    alive = 0 if int(ram[PLAYER_STATE]) in DEAD_STATES else 1
    return dict(
        world_x=world_x,
        world_y=world_y,
        y_screen=int(ram[y_fine]),
        scroll=scroll,
        alive=alive,
        power=int(ram[POWER_STATE]),
    )


# ---- invariants a real walk-right trace must satisfy ------------------------
@dataclass
class Invariant:
    field: str
    label: str
    fn: object   # (list[int] series, dict context) -> (bool, str)


def _inv_x_monotone(series, ctx):
    """world_x must climb ~1-2 px per emulator frame and never wrap backwards."""
    if len(series) < 10:
        return False, "trace too short"
    deltas = [b - a for a, b in zip(series, series[1:])]
    fwd = sum(1 for d in deltas if d > 0)
    backjump = sum(1 for d in deltas if d < -50)      # page-wrap corruption
    rate = (series[-1] - series[0]) / max(1, len(series) - 1)
    ok = fwd >= 0.7 * len(deltas) and backjump == 0 and 0.5 <= rate <= 8.0
    return ok, f"rise {series[0]}->{series[-1]}, {rate:.2f}px/obs, {fwd}/{len(deltas)} fwd, {backjump} backjumps"


def _inv_scroll_tracks_x(series, ctx):
    """Once Mario passes mid-screen the camera scrolls, so scroll must track x."""
    x = ctx.get("world_x") or []
    if len(x) != len(series) or len(x) < 10:
        return False, "no paired world_x"
    moved = series[-1] - series[0]
    xmoved = x[-1] - x[0]
    if xmoved <= 0:
        return False, "x did not advance; fix world_x first"
    ratio = moved / xmoved
    ok = 0.3 <= ratio <= 1.2 and moved > 0
    return ok, f"scroll +{moved} vs x +{xmoved} (ratio {ratio:.2f})"


def _inv_y_sane(series, ctx):
    """
    y must sit in a plausible band and MOVE during a jump. A wrong address is
    usually either constant (dead byte) or wildly out of range.
    """
    if len(series) < 10:
        return False, "trace too short"
    lo, hi = min(series), max(series)
    spread = hi - lo
    ok = spread >= 8 and 0 <= lo and hi < 2048
    return ok, f"range {lo}..{hi} (spread {spread})"


INVARIANTS = [
    Invariant("world_x", "climbs ~1-2px/frame, no page-wrap backjumps", _inv_x_monotone),
    Invariant("scroll",  "tracks world_x past mid-screen",              _inv_scroll_tracks_x),
    Invariant("world_y", "plausible range and moves during a jump",     _inv_y_sane),
]
