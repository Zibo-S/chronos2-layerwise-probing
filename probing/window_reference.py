"""Window-identity references: making "the models saw the same windows" checkable for datasets
that have no committed Chronos-2 run.

THE PROBLEM THIS FIXES. Cross-model claims rest on every model being probed on byte-identical
windows. For the original seven that is verifiable: ``results/ext_v5_native_head_adapter/``
holds the committed Chronos-2 configs and per-window test series ids, and
``assert_window_parity`` compares against them element-wise.

For a dataset added after that run there is nothing to compare against. The old code returned
``None`` from ``chronos_reference()`` and the audit printed ``no_committed_reference`` with
``parity_ok=None`` — which reads like a pass and is not one. A silent "no reference" is the
worst outcome: the run proceeds, the number lands in a table, and nobody can later tell whether
the second model was evaluated on the same windows or not.

THE WORKFLOW. Creating a reference is now a DELIBERATE, separate act:

  1. ``--reference-mode require`` (the default) — a dataset with no reference ABORTS, naming the
     exact command that would create one. Nothing is fitted, no GPU time is spent.
  2. ``--reference-mode create`` — this run writes the reference for any dataset that lacks one,
     then proceeds. It REFUSES to overwrite an existing reference (that would silently redefine
     what parity means); changing one takes ``--force-reference`` and is recorded.
  3. every later run, in any model line, finds that reference and checks against it exactly as
     the original seven are checked against the committed Chronos-2 artifacts.
  4. ``--reference-mode allow-missing`` — the old permissive behavior, now something you have to
     ask for by name, for exploratory runs.

WHAT A REFERENCE PINS. Counts (train/val/test), the geometry (C, H), the seasonal period, the
deterministic series-cap audit, the per-window test series/cluster ids element-wise, and a
digest of the test contexts themselves. The digest is the strongest check: it catches a
windowing change that happens to preserve the ids, which the committed Chronos-2 artifacts
cannot detect because they never stored the contexts.

A reference is a claim about WINDOWS ONLY. It says nothing about which model, probe, quantile
set or seed produced any number.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from probing import registry
from probing.config import REPO_ROOT

__all__ = ["REFERENCE_ROOT", "REFERENCE_MODES", "SCHEMA", "reference_paths",
           "window_digest", "build_reference", "write_reference", "read_reference",
           "resolve_reference", "compare_to_reference", "MissingReferenceError"]

SCHEMA = "window_reference/v1"
REFERENCE_ROOT = REPO_ROOT / "results" / "window_references"
REFERENCE_MODES = ("require", "create", "allow-missing")


class MissingReferenceError(RuntimeError):
    """Raised under ``--reference-mode require`` when a dataset has no window reference."""


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _rel(p: Path) -> str:
    """Repo-relative path when possible, absolute otherwise (tests and --reference-root may
    point outside the repo, and a display helper must never be the thing that raises)."""
    try:
        return str(Path(p).relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def reference_paths(tag: str, root: Path | None = None) -> dict[str, Path]:
    root = Path(root) if root is not None else REFERENCE_ROOT
    return {"json": root / f"window_reference__{tag}.json",
            "npz": root / f"window_reference__{tag}.npz"}


def window_digest(w) -> str:
    """A content hash of the test windows themselves — contexts, targets, ids and origins.

    Hashing the CONTEXTS (not just their count and series ids) is what makes this able to catch
    a re-windowing that preserves the identifiers: a different stride, a changed validity
    filter or a shifted origin all move these bytes. float32 arrays are hashed in their exact
    stored representation, so the digest is reproducible across machines.
    """
    h = hashlib.sha256()
    for key in ("X_test", "Y_test_traj", "series_test"):
        a = np.ascontiguousarray(w[key])
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    origins = (w["meta"].get("origins") or {}).get("test")
    if origins is not None:
        h.update(b"origins")
        h.update(np.ascontiguousarray(np.asarray(origins, np.int64)).tobytes())
    return "sha256:" + h.hexdigest()


def build_reference(tag: str, w, *, created_by: str) -> dict:
    """The reference record for one dataset's windows (JSON-serializable part)."""
    m = w["meta"]
    sid = np.asarray(w["series_test"], np.int64)
    return {
        "schema": SCHEMA,
        "dataset": tag,
        "display_name": registry.display_name(tag),
        "builder": registry.builder(tag),
        "cluster_unit": m.get("cluster_unit", registry.cluster_unit(tag)),
        "split_mode": m.get("split_mode"),
        "C": m.get("C"), "H": m.get("H"), "seed": m.get("seed"),
        "seasonal_m": m.get("m_season"),
        "n_train": int(len(w["X_train"])),
        "n_val": int(len(w["X_val"])) if "X_val" in w else None,
        "n_test": int(len(sid)),
        "n_test_units": int(np.unique(sid).size),
        "series_cap": m.get("series_cap"),
        "window_digest": window_digest(w),
        "created_by": created_by,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
    }


def write_reference(tag: str, w, *, created_by: str, root: Path | None = None,
                    force: bool = False) -> dict:
    """Write the reference for ``tag``. Refuses to overwrite unless ``force``.

    Overwriting silently would redefine what "same windows" means for every past and future
    run, which is exactly the thing a reference exists to prevent.
    """
    paths = reference_paths(tag, root)
    if paths["json"].exists() and not force:
        raise FileExistsError(
            f"a window reference for {tag!r} already exists at "
            f"{_rel(paths['json'])}. Overwriting it would silently redefine "
            f"what window parity means for every run that has already been checked against it. "
            f"Pass --force-reference if you really intend to redefine it.")
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    rec = build_reference(tag, w, created_by=created_by)
    sid = np.asarray(w["series_test"], np.int64)
    origins = (w["meta"].get("origins") or {}).get("test")
    arrays = {"series_test": sid}
    if origins is not None:
        arrays["origins_test"] = np.asarray(origins, np.int64)
    np.savez(paths["npz"], **arrays)
    paths["json"].write_text(json.dumps(rec, indent=2))
    rec["paths"] = {k: _rel(v) for k, v in paths.items()}
    return rec


def read_reference(tag: str, root: Path | None = None) -> dict | None:
    """Read a previously written reference, or None when there is none."""
    paths = reference_paths(tag, root)
    if not (paths["json"].exists() and paths["npz"].exists()):
        return None
    rec = json.loads(paths["json"].read_text())
    if rec.get("schema") != SCHEMA:
        raise RuntimeError(f"{paths['json']}: schema {rec.get('schema')!r} != {SCHEMA!r}")
    with np.load(paths["npz"], allow_pickle=False) as z:
        sid = np.asarray(z["series_test"], np.int64)
    return {"kind": "window_reference", "config": rec, "series_test": sid,
            "paths": {k: _rel(v) for k, v in paths.items()}}


def resolve_reference(tag: str, committed_lookup, root: Path | None = None) -> dict | None:
    """The reference to check ``tag`` against: the committed Chronos-2 artifact if one exists,
    otherwise a written window reference, otherwise None.

    Committed artifacts win so the original seven keep being checked against exactly what they
    were checked against before — a written reference can never shadow them.
    """
    ref = committed_lookup(tag)
    if ref is not None:
        ref.setdefault("kind", "committed_chronos2")
        return ref
    return read_reference(tag, root)


def compare_to_reference(tag: str, w, ref: dict) -> list[str]:
    """Element-wise comparison against a written window reference. Returns failure strings.

    Only used for ``kind == "window_reference"``; the committed Chronos-2 artifacts keep their
    own comparison in the driver, unchanged.
    """
    c = ref["config"]
    m = w["meta"]
    sid = np.asarray(w["series_test"], np.int64)
    rsid = np.asarray(ref["series_test"], np.int64)
    fails = []
    for name, got, want in (("C", m.get("C"), c["C"]),
                            ("H", m.get("H"), c["H"]),
                            ("seasonal_m", m.get("m_season"), c["seasonal_m"]),
                            ("builder", registry.builder(tag), c["builder"]),
                            ("n_train", int(len(w["X_train"])), c["n_train"]),
                            ("n_test", int(len(sid)), c["n_test"])):
        if got != want:
            fails.append(f"{name}: this run={got!r} vs reference={want!r}")
    if "X_val" in w and c.get("n_val") is not None and int(len(w["X_val"])) != c["n_val"]:
        fails.append(f"n_val: this run={len(w['X_val'])} vs reference={c['n_val']}")
    if sid.shape != rsid.shape:
        fails.append(f"series_test shape: {sid.shape} vs {rsid.shape}")
    elif not np.array_equal(sid, rsid):
        d = int((sid != rsid).sum())
        first = int(np.flatnonzero(sid != rsid)[0])
        fails.append(f"series_test differs in {d}/{len(sid)} windows (first at index {first}: "
                     f"{int(sid[first])} vs {int(rsid[first])})")
    got_digest = window_digest(w)
    if got_digest != c["window_digest"]:
        fails.append(f"window_digest: {got_digest} vs reference {c['window_digest']} — the test "
                     f"CONTEXTS themselves differ, not just their identifiers")
    return fails


def missing_reference_message(tag: str, created_by: str) -> str:
    """The abort text for ``--reference-mode require``: says exactly how to create one."""
    rel = _rel(reference_paths(tag)["json"])
    return (
        f"NO WINDOW REFERENCE for {tag!r}.\n"
        f"  There is no committed Chronos-2 artifact for this dataset and no window reference "
        f"at {rel}.\n"
        f"  Without one, 'the models saw the same windows' is an assumption, not a check — so "
        f"this run refuses to proceed rather than print an unverifiable number.\n"
        f"  To create the reference deliberately, re-run {created_by} with:\n"
        f"      --reference-mode create --datasets {tag}\n"
        f"  (or --reference-mode allow-missing to run without any parity check, which must not "
        f"be used for anything reported).")
