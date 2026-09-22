"""THE cross-model window-parity check: one implementation, called by every model line.

A cross-model claim ("Chronos-2's tunnel opens earlier than TiRex's") is only meaningful if the
three models were asked the same question — i.e. scored on byte-identical
``x[t-512:t] -> x[t:t+64]`` windows. This module is where that stops being an assumption.

WHY IT LIVES HERE AND NOT IN A DRIVER. The implementation grew inside
``experiments/run_timesfm3_last_token_probing.py``, where it was correct but unreachable: the
TiRex line and the new Phase-1 orchestrator would each have had to re-type it, and a re-typed
comparison is a comparison that can drift. The reference lookup, the reference-mode policy and
the exact failure wording now live in exactly one place.

WHAT IS CHECKED, against the reference:
  * C, H and the seasonal period;
  * the train / val / test window COUNTS;
  * the per-window test series/cluster ids, ELEMENT-WISE (the decisive check);
  * for written references, additionally a sha256 digest of the test CONTEXTS themselves —
    which catches a re-windowing that happens to preserve the ids, something the committed
    Chronos-2 artifacts cannot detect because they never stored the contexts.

TWO KINDS OF REFERENCE, and committed artifacts always win:
  ``committed_chronos2``  results/ext_v5_native_head_adapter/{configs,bootstrap_inputs} — the
                          original seven, checked against exactly what they were always checked
                          against;
  ``window_reference``    results/window_references/window_reference__<tag>.{json,npz} — written
                          deliberately (see ``probing.window_reference``) for a dataset with no
                          committed Chronos-2 run.

NO PRETRAINING LABEL IS EMITTED HERE. The identity record carries the dataset's registry facts
(display name, role, seasonal period, builder, cluster unit) and nothing about whose corpus the
dataset appeared in. The legacy flat PT-ID/PT-OOD string is a CALLER's decoration, passed in via
``extra_ident`` by the committed Chronos-2-era driver that already emits it; the Phase-1 pipeline
passes nothing, so no Phase-1 artifact can contain a global pretraining classification. Model-
relative provenance is ``registry.provenance(tag, model)`` and lives in the cell record instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from probing import registry
from probing import window_reference as wref
from probing.config import REPO_ROOT

__all__ = ["CHRONOS_REF_ROOT", "chronos_reference", "window_reference_for", "window_identity",
           "assert_window_parity", "parity_for"]

#: The committed Chronos-2 run whose windows every other model line is compared against.
CHRONOS_REF_ROOT = REPO_ROOT / "results" / "ext_v5_native_head_adapter"


# --------------------------------------------------------------------------- #
# reference lookup
# --------------------------------------------------------------------------- #
def chronos_reference(tag: str, root=CHRONOS_REF_ROOT):
    """The committed Chronos-2 artifacts for ``tag``: window counts + per-window test series ids.

    configs/native_head_adapter__<tag>__config.json  -> n_train / n_val / n_test / C / H / kind
    bootstrap_inputs/native_head_adapter__<tag>.npz  -> series_test, one id per test window
    Returns None when the dataset is not part of that committed run (every dataset added after
    it, plus the legacy extended_v1 tags).
    """
    cfg = Path(root) / "configs" / f"native_head_adapter__{tag}__config.json"
    npz = Path(root) / "bootstrap_inputs" / f"native_head_adapter__{tag}.npz"
    if not (cfg.exists() and npz.exists()):
        return None
    c = json.loads(cfg.read_text())
    with np.load(npz, allow_pickle=False) as z:
        sid = np.asarray(z["series_test"], np.int64)
    return {"config": c, "series_test": sid,
            "paths": {"config": str(cfg.relative_to(REPO_ROOT)),
                      "bootstrap_inputs": str(npz.relative_to(REPO_ROOT))}}


def window_reference_for(tag, root=None):
    """The reference ``tag`` is checked against, or None.

    Committed Chronos-2 artifacts win over written references, so the original seven keep being
    compared against exactly what they always were and a written reference can never shadow one.
    """
    return wref.resolve_reference(tag, chronos_reference, root)


# --------------------------------------------------------------------------- #
# identity record
# --------------------------------------------------------------------------- #
def window_identity(tag, w, extra: dict | None = None):
    """The reportable identity of one dataset's windows (counts + the first few identifiers).

    ``extra`` is merged in verbatim and is the ONLY way a caller-specific field (e.g. the legacy
    Chronos-2 PT-ID/PT-OOD string) enters the record — this function never invents one.
    """
    sid = np.asarray(w["series_test"], np.int64)
    origins = (w["meta"].get("origins", {}) or {}).get("test")
    ids = ([f"s{int(a)}@t{int(b)}" for a, b in zip(sid[:6], origins[:6])] if origins
           else [f"s{int(a)}" for a in sid[:6]])
    ident = {"dataset": tag, "short": registry.display_name(tag),
             "role": registry.role(tag),
             "seasonal_m": registry.seasonal_m(tag), "builder": registry.builder(tag),
             "split_mode": w["meta"].get("split_mode"),
             "n_train_windows": int(len(w["X_train"])),
             "n_val_windows": int(len(w["X_val"])) if "X_val" in w else None,
             "n_test_windows": int(len(sid)),
             "n_test_series": int(len(np.unique(sid))),
             "cluster_unit": w["meta"].get("cluster_unit", registry.cluster_unit(tag)),
             "first_test_identifiers": ids,
             "test_series_first6": [int(x) for x in sid[:6]]}
    if extra:
        ident.update(extra)
    return ident


# --------------------------------------------------------------------------- #
# the check
# --------------------------------------------------------------------------- #
def assert_window_parity(tag, w, ref, *, strict=True, reference_mode="require",
                         created_by="a model driver", reference_root=None,
                         force_reference=False, extra_ident: dict | None = None,
                         model_label="this run"):
    """Every model must receive the SAME windows. Raises on any mismatch unless ``strict=False``.

    ``reference_mode`` decides what a MISSING reference means. It used to mean "carry on with
    parity_ok=None", which reads like a pass; a dataset added after the committed Chronos-2 run
    could therefore be probed by a second model with nothing checking the windows at all.
      require       -- abort, naming the command that would create the reference (default);
      create        -- write the reference from THESE windows, then proceed;
      allow-missing -- the old permissive behavior, now opt-in and never for reported numbers.

    ``model_label`` only appears in failure text ("TimesFM-3=... vs Chronos-2=..."); it changes
    no comparison.
    """
    ident = window_identity(tag, w, extra_ident)
    if ref is None:
        if reference_mode == "require":
            raise wref.MissingReferenceError(wref.missing_reference_message(tag, created_by))
        if reference_mode == "create":
            rec = wref.write_reference(tag, w, created_by=created_by, root=reference_root,
                                       force=force_reference)
            print(f"  [window reference CREATED] {rec['paths']['json']}\n"
                  f"      digest {rec['window_digest']}  -- every later run of any model line "
                  f"is now checked against these windows")
            ident.update(chronos_parity="reference_created", parity_ok=None,
                         reference_kind="window_reference", reference_created=True,
                         window_reference=rec["paths"])
            return ident
        if reference_mode == "allow-missing":
            ident.update(chronos_parity="NO REFERENCE (unchecked)", parity_ok=None,
                         reference_kind=None,
                         parity_warning="windows are UNVERIFIED against any other model; this "
                                        "run must not be used for a cross-model claim")
            print(f"  [WARNING] {tag}: no window reference and --reference-mode allow-missing; "
                  f"cross-model window parity is UNCHECKED for this dataset")
            return ident
        raise ValueError(f"unknown reference mode {reference_mode!r}; "
                         f"known: {wref.REFERENCE_MODES}")

    if ref.get("kind") == "window_reference":
        fails = wref.compare_to_reference(tag, w, ref)
        ident.update(chronos_parity="match" if not fails else "MISMATCH",
                     parity_ok=not fails, parity_failures=fails,
                     reference_kind="window_reference", window_reference=ref["paths"])
        if fails and strict:
            raise RuntimeError(
                f"WINDOW PARITY FAILED for {tag}: these windows differ from the reference "
                f"every other run was checked against.\n    " + "\n    ".join(fails) +
                f"\n  Reference: {ref['paths']['json']}.\n"
                "  Fix the window construction -- do NOT proceed with mismatched windows.")
        return ident

    c, m = ref["config"], w["meta"]
    sid = np.asarray(w["series_test"], np.int64)
    rsid = ref["series_test"]
    fails = []
    checks = [("C", m.get("C"), c["C"]), ("H", m.get("H"), c["H"]),
              ("seasonal_m", m.get("m_season"), c["seasonal_m"]),
              ("n_train", int(len(w["X_train"])), c["n_train"]),
              ("n_test", int(len(sid)), c["n_test"])]
    if extra_ident and "kind" in extra_ident and "kind" in c:
        checks.append(("kind", extra_ident["kind"], c["kind"]))
    for name, got, want in checks:
        if got != want:
            fails.append(f"{name}: {model_label}={got!r} vs Chronos-2={want!r}")
    if "X_val" in w and int(len(w["X_val"])) != c["n_val"]:
        fails.append(f"n_val: {model_label}={len(w['X_val'])} vs Chronos-2={c['n_val']}")
    if sid.shape != rsid.shape:
        fails.append(f"series_test shape: {sid.shape} vs {rsid.shape}")
    elif not np.array_equal(sid, rsid):
        d = int((sid != rsid).sum())
        first = int(np.flatnonzero(sid != rsid)[0])
        fails.append(f"series_test differs in {d}/{len(sid)} windows (first at index {first}: "
                     f"{int(sid[first])} vs {int(rsid[first])})")
    ident.update(chronos_parity="match" if not fails else "MISMATCH",
                 parity_ok=not fails, parity_failures=fails,
                 reference_kind="committed_chronos2", chronos_reference=ref["paths"],
                 chronos_counts={"n_train": c["n_train"], "n_val": c["n_val"],
                                 "n_test": c["n_test"], "dataset_set": c["dataset_set"]})
    if fails and strict:
        raise RuntimeError(
            f"WINDOW PARITY FAILED for {tag}: {model_label} is not being evaluated on the same "
            f"windows as Chronos-2.\n    " + "\n    ".join(fails) +
            f"\n  Reference: {ref['paths']['config']} + {ref['paths']['bootstrap_inputs']}.\n"
            "  Fix the window construction -- do NOT proceed with mismatched windows.")
    return ident


def parity_for(tag, w, args, created_by, strict=None, extra_ident=None, model_label="this run"):
    """THE parity entry point every model driver calls.

    One function so the reference lookup, the reference-mode policy and the failure wording can
    never drift between the Chronos-2, TimesFM-3, TiRex and Phase-1 lines. ``getattr`` defaults
    mean a driver that has not yet grown the CLI flags still gets the SAFE behavior
    (``require``) rather than the permissive one.
    """
    return assert_window_parity(
        tag, w, window_reference_for(tag, getattr(args, "reference_root", None)),
        strict=(not getattr(args, "allow_window_mismatch", False)) if strict is None else strict,
        reference_mode=getattr(args, "reference_mode", "require"),
        created_by=created_by,
        reference_root=getattr(args, "reference_root", None),
        force_reference=getattr(args, "force_reference", False),
        extra_ident=extra_ident, model_label=model_label)
