"""
Stage 3 adapter: train SAEs on JEPA latents with aquin's SAE tooling.

WHY THIS FILE EXISTS
--------------------
aquin's activation store is genuinely dimension-agnostic: a directory of
`chunk_*.pt`, each a (N, d) tensor, optional `norm.pt` with {mean,std}, optional
`manifest.json`. validate_acts_manifest() raises ONLY on a layer mismatch — a
model-id mismatch is a printed warning, and a missing manifest short-circuits
validation entirely. So 192-dim JEPA latents go in cleanly.

BUT the `aquin sae train` CLI cannot be used: cmd_sae_train never passes
d_model down, so the SAE is built at the session model's width (768 for GPT-2)
and silently mismatches our 192. We therefore call the Python entrypoint
directly, where d_model and n_features are real keyword arguments.

Two warts, accepted knowingly rather than papered over:
  * `model_id` must resolve in aquin's catalog (resolve_model_id runs before the
    d_model override), so we pass a placeholder purely for config lookup. The
    trained SAE registers under that placeholder name.
  * `layer` is unchecked when no manifest is written, so we reuse it as a free
    integer slot to key the SIGReg-lambda index.
"""
from __future__ import annotations

from pathlib import Path

from aqmario.config import CFG


def dump_latents(ckpt: str, out_dir: str | Path, loader=None,
                 chunk_vectors: int | None = None) -> Path:
    """
    Encode frames -> 192-dim latents and write them as aquin activation chunks.

    Deliberately writes NO manifest.json: with none present aquin skips model-id
    and layer validation entirely, which is exactly what we want for a latent
    space that has no catalog model behind it.
    """
    import torch

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    n = chunk_vectors or CFG.sae.chunk_vectors

    if loader is None:
        raise NotImplementedError(
            "dump_latents needs a loader yielding batches of frames. "
            "Wire it to the Stage 2 encoder in scripts/run_gates.py.")

    buf, idx, total = [], 0, 0
    for z in loader:                      # z: (B, 192) encoder output
        buf.append(z.detach().cpu().float())
        if sum(b.shape[0] for b in buf) >= n:
            t = torch.cat(buf)[:n]
            torch.save(t, out / f"chunk_{idx:05d}.pt")
            total += t.shape[0]
            buf, idx = [torch.cat(buf)[n:]], idx + 1
    if buf and (rest := __import__("torch").cat(buf)).shape[0]:
        torch.save(rest, out / f"chunk_{idx:05d}.pt")
        total += rest.shape[0]

    # norm.pt is optional but saves aquin a full pass to compute mean/std
    import torch as T
    allv = T.cat([T.load(p, map_location="cpu") for p in sorted(out.glob("chunk_*.pt"))])
    T.save({"mean": allv.mean(0), "std": allv.std(0).clamp_min(1e-6)}, out / "norm.pt")
    print(f"[aq_sae] wrote {idx+1} chunks, {total} vectors of dim {allv.shape[1]} -> {out}")
    if total < 10_000:
        print("[aq_sae] warning: aquin warns below 10k vectors; SAE quality will be poor")
    return out


def train_latent_sae(acts_dir: str | Path, tag: str, layer: int = 0,
                     output: str | Path | None = None,
                     d_model: int | None = None,
                     n_features: int | None = None,
                     max_steps: int | None = None) -> Path:
    """
    Call aquin's SAE trainer on our latents with an explicit d_model.
    This is the path the CLI cannot express.
    """
    from aquin.compute.sae_train import train_sae

    out = Path(output) if output else (CFG.run_dir / "sae" / f"{tag}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    return train_sae(
        CFG.sae.placeholder_model_id,       # catalog lookup only; overridden below
        layer,
        out,
        d_model=d_model or CFG.sae.d_model,          # 192, NOT the catalog width
        n_features=n_features or CFG.sae.n_features,  # 2048, not aquin's 32768 default
        max_steps=max_steps,
        activations_dir=str(acts_dir),                # implies no forward passes
    )


def lambda_diff(sae_paths: dict, output: str | Path | None = None) -> Path:
    """
    The Stage 3 headline: which features does SIGReg lambda=0.5 destroy?
    `aquin sae align` compares two SAE files; both are 192-dim here, so this
    part of the toolchain works unmodified.
    """
    import itertools, json, subprocess

    out = Path(output) if output else (CFG.run_dir / "sae" / "lambda_diff.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs, results = list(itertools.combinations(sorted(sae_paths), 2)), []
    for a, b in pairs:
        r = subprocess.run(["aquin", "sae", "align",
                            "--sae-a", str(sae_paths[a]), "--sae-b", str(sae_paths[b])],
                           capture_output=True, text=True)
        results.append({"lambda_a": a, "lambda_b": b, "ok": r.returncode == 0,
                        "stdout": r.stdout[-4000:], "stderr": r.stderr[-1000:]})
    out.write_text(json.dumps({"pairs": results, "features": []}, indent=2))
    print(f"[aq_sae] wrote {out}")
    return out
