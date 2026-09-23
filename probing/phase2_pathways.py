"""Phase-2 native forecasting pathways + READ-ONLY access to the Phase-1 feature caches.

For each model the native pathway is f = g o n followed by the model's own inverse transform:
n = the final normalization (identity if none), g = the pretrained output head. An intermediate
state h_l is pushed through f unchanged (the HARD CUT) or after an adapter (ALIGNMENT). The
pathway modules are the CHECKPOINT'S OWN modules, frozen and in eval mode -- never
re-implemented -- and gradients flow through them only to a trainable adapter.

    model      rows / window        n                          g                              inverse
    Chronos-2  4 forecast slots     encoder.final_layer_norm   output_patch_embedding         mu + sd * sinh(z)
                                    (T5-style RMS, no mean,    ResidualBlock 768->3072->21*16 (context
                                    no bias)                   layout "b n (q p) -> b q (n p)" instance norm)
    TimesFM-3  token 15 (1 row)     none                       output_head Linear(1280, 64*9) z*sd+mu -> clamp
                                                               HORIZON-major (t*Q + q)        -> + trend
    TiRex      token 63 of each     out_norm (RMSNorm, fp32)   output_patch_embedding         per-pass
               of the 2 passes                                 ResidualBlock 512->2048->9*32  z*scale_k + loc_k
                                                               QUANTILE-major (q*P + t)

Every array handed to a Phase-2 fit comes from a Phase-1 cache that ALREADY EXISTS. The loaders
here never extract: a cache miss is a hard error naming the Phase-1 step that should have produced
it, because Phase 1 is frozen and its caches are read-only for Phase 2 (and were written
non-atomically, so Phase 2 must never become a second writer).

Rows are the Phase-1 rows: all windows for Chronos-2 and TiRex, the ``valid`` windows for
TimesFM-3 (sigma >= SIGMA_EPS) -- and the per-split cluster ids are checked element-wise against
the ones the Phase-1 cell saved, so an H3 number and a Phase-1 number always describe the same
observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from probing import phase1
from probing.phase2 import MEDIAN_INDEX, QUANTILES, Phase1DependencyError

__all__ = ["SplitData", "CellData", "NativePathway", "Chronos2Pathway", "TimesFM3Pathway",
           "TiRexPathway", "chronos2_cache_candidates", "load_chronos2", "load_timesfm3",
           "load_tirex", "check_cluster_ids", "SPLITS"]

SPLITS = ("train", "val", "test")
H = phase1.PHASE1_H
C = phase1.PHASE1_C


@dataclass
class SplitData:
    """One split of one cell, restricted to the Phase-1 rows."""
    feats: dict                       # depth-axis index -> (n, R, d) float32
    target: np.ndarray                # (n, H) float32, the Phase-1 normalized target
    y_raw: np.ndarray                 # (n, H) float64, raw future
    X: np.ndarray                     # (n, C) float32, raw contexts
    inv: dict                         # model-specific inverse statistics
    cluster_ids: np.ndarray           # (n,) bootstrap unit
    rows: np.ndarray                  # (n,) indices into the window split
    head_input: np.ndarray | None = None      # cached post-norm final state (C2 L12+LN, TiRex L12+RMS)
    native_raw: np.ndarray | None = None      # (n, H, 9) Phase-1 cached native forecast (gate only)


@dataclass
class CellData:
    model: str
    tag: str
    splits: dict
    depth_labels: list
    reference_index: int
    provenance: dict = field(default_factory=dict)

    @property
    def depth_indices(self) -> list[int]:
        return list(range(len(self.depth_labels)))


# --------------------------------------------------------------------------- #
# pathways
# --------------------------------------------------------------------------- #
def _freeze(*mods):
    for m in mods:
        if m is None:
            continue
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)


class NativePathway:
    """Common interface. ``outputs`` is what NOA matches; ``q9`` is what every metric scores."""
    model = ""
    d = 0
    rows_per_window = 1
    n_out = 0

    def outputs(self, h: torch.Tensor) -> torch.Tensor:           # (n, R, d) -> (n, R, n_out)
        raise NotImplementedError

    def q9(self, out: torch.Tensor) -> torch.Tensor:              # (n, R, n_out) -> (n, 9, H)
        raise NotImplementedError

    def to_raw(self, z9, split: SplitData) -> np.ndarray:         # (n, 9, H) raw units
        raise NotImplementedError

    def modules(self) -> list:
        raise NotImplementedError

    @property
    def device(self):
        return next(self.modules()[-1].parameters()).device

    def assert_frozen(self) -> int:
        n = 0
        for m in self.modules():
            for name, p in m.named_parameters():
                if p.requires_grad:
                    raise RuntimeError(f"{self.model} pathway parameter {name} requires grad")
                n += p.numel()
            if m.training:
                raise RuntimeError(f"{self.model} pathway module {type(m).__name__} is in train "
                                   "mode (dropout would make the pathway stochastic)")
        return n

    def linear_head(self):
        return None

    @torch.no_grad()
    def run(self, feats: np.ndarray, adapter=None, batch_size: int = 4096) -> tuple:
        """Frozen forward on numpy states -> (normalized head outputs, (n, 9, H) numpy float64)."""
        outs, zs = [], []
        dev = self.device
        for s in range(0, len(feats), batch_size):
            h = torch.as_tensor(np.ascontiguousarray(feats[s:s + batch_size]), dtype=torch.float32,
                                device=dev)
            if adapter is not None:
                h = adapter(h)
            o = self.outputs(h)
            outs.append(o.float().cpu())
            zs.append(self.q9(o).double().cpu())
        return torch.cat(outs).numpy(), torch.cat(zs).numpy()

    def describe(self) -> dict:
        return {"model": self.model, "d": self.d, "rows_per_window": self.rows_per_window,
                "n_out_per_row": self.n_out}


class Chronos2Pathway(NativePathway):
    model = "chronos2"
    d = 768
    K = 4
    P = 16

    def __init__(self, head, norm, native_quantiles):
        self.head, self.norm = head, norm
        _freeze(head, norm)
        nq = np.asarray(native_quantiles, np.float64)
        self.n_native_q = len(nq)
        self.n_out = self.n_native_q * self.P
        idx = []
        for q in QUANTILES:
            hit = np.flatnonzero(np.abs(nq - q) < 1e-6)
            if hit.size != 1:
                raise RuntimeError(f"Chronos-2 native quantiles {nq.tolist()} do not contain {q} "
                                   "exactly once; Q9 would have to be interpolated")
            idx.append(int(hit[0]))
        self.q9_index = idx
        self._q9_idx_t = None
        out_features = head.output_layer.out_features if hasattr(head, "output_layer") else None
        if out_features is not None and out_features != self.n_out:
            raise RuntimeError(f"native head emits {out_features} per slot, expected "
                               f"{self.n_native_q}*{self.P}")

    @classmethod
    def from_pipeline(cls, pipeline):
        from probing.native_head_adapter import native_head_modules
        head, norm = native_head_modules(pipeline)
        return cls(head, norm, list(pipeline.quantiles))

    def modules(self):
        return [self.norm, self.head]

    def outputs(self, h):
        return self.head(self.norm(h))

    def q_all(self, out):
        n = out.shape[0]
        return (out.view(n, self.K, self.n_native_q, self.P).permute(0, 2, 1, 3)
                .reshape(n, self.n_native_q, self.K * self.P)[:, :, :H])

    def q9(self, out):
        if self._q9_idx_t is None or self._q9_idx_t.device != out.device:
            self._q9_idx_t = torch.as_tensor(self.q9_index, device=out.device)
        return self.q_all(out).index_select(1, self._q9_idx_t)

    def to_raw(self, z9, split):
        z = np.asarray(z9, np.float64)
        mu, sd = split.inv["mu"], split.inv["sd"]
        return mu[:, None, None] + sd[:, None, None] * np.sinh(z)

    def describe(self):
        return {**super().describe(), "native_quantiles": self.n_native_q,
                "q9_native_indices": self.q9_index,
                "layout": "b n (q p) -> b q (n p)  (Chronos2Model.forward)",
                "inverse": "loc + scale * sinh(z)  (InstanceNorm, use_arcsinh=True)",
                "final_norm": "encoder.final_layer_norm: T5-style RMS scaling, no mean, no bias"}


class TimesFM3Pathway(NativePathway):
    model = "timesfm3"
    d = 1280
    rows_per_window = 1

    def __init__(self, head, value_clip: float, output_patch_len: int, num_quantiles: int):
        self.head = head
        _freeze(head)
        self.value_clip = float(value_clip)
        self.opl, self.nq = int(output_patch_len), int(num_quantiles)
        self.n_out = self.opl * self.nq
        if self.nq != len(QUANTILES):
            raise RuntimeError(f"TimesFM-3 head has {self.nq} quantiles, expected 9")
        if head.out_features != self.n_out or head.in_features != self.d:
            raise RuntimeError(f"output_head is {head.in_features}->{head.out_features}")

    @classmethod
    def from_model(cls, model):
        return cls(model.output_head, model.value_clip, model.output_patch_len,
                   model.num_quantiles)

    def modules(self):
        return [self.head]

    def outputs(self, h):
        return self.head(h)

    def q9(self, out):
        n = out.shape[0]
        return out.reshape(n, self.opl, self.nq)[:, :H, :].permute(0, 2, 1)

    def to_raw(self, z9, split):
        z = np.asarray(z9, np.float64)
        mu, sd, trend = split.inv["mu"], split.inv["sd"], split.inv["trend"]
        den = z * sd[:, None, None] + mu[:, None, None]
        den = np.clip(den, -self.value_clip, self.value_clip)
        return den + trend[:, None, :]

    def linear_head(self):
        W = self.head.weight.detach().double().cpu().numpy()
        c = self.head.bias.detach().double().cpu().numpy()
        return W, c

    def describe(self):
        return {**super().describe(), "layout": "horizon-major (t*Q + q)",
                "inverse": "revin reverse (z*sd + mu) -> clamp(+-value_clip) -> + trend "
                           "(decode's own path; stitch is the identity for one output token)",
                "value_clip": self.value_clip, "final_norm": "none"}


class TiRexPathway(NativePathway):
    model = "tirex"
    d = 512
    K = 2

    def __init__(self, head, norm, geom):
        self.head, self.norm, self.geom = head, norm, geom
        _freeze(head, norm)
        self.nq, self.P = int(geom.num_quantiles), int(geom.output_patch)
        self.K = int(geom.n_forecast_patches)
        self.n_out = self.nq * self.P
        if self.nq != len(QUANTILES):
            raise RuntimeError(f"TiRex head has {self.nq} quantiles, expected 9")

    @classmethod
    def from_model(cls, model, geom):
        return cls(model.output_patch_embedding, model.out_norm, geom)

    def modules(self):
        return [self.norm, self.head]

    @property
    def rows_per_window(self):
        return self.K

    def outputs(self, h):
        return self.head(self.norm(h))

    def q9(self, out):
        n = out.shape[0]
        return (out.view(n, self.K, self.nq, self.P).permute(0, 2, 1, 3)
                .reshape(n, self.nq, self.K * self.P)[:, :, :H])

    def to_raw(self, z9, split):
        from probing.tirex_model import denormalize
        z = np.asarray(z9, np.float64)
        loc, scale = split.inv["loc"], split.inv["scale"]
        return np.stack([denormalize(z[:, q, :], loc, scale, self.geom)
                         for q in range(z.shape[1])], axis=1)

    def describe(self):
        return {**super().describe(), "layout": "quantile-major (q*P + t), passes concatenated",
                "inverse": "per-pass z*scale_k + loc_k (StandardScaler.re_scale)",
                "final_norm": "out_norm: RMSNorm computed in fp32", "passes": self.K}


# --------------------------------------------------------------------------- #
# read-only loaders
# --------------------------------------------------------------------------- #
def check_cluster_ids(model: str, tag: str, split: str, ours, phase1_arrays: dict) -> None:
    key = f"cluster_ids_{split}"
    if key not in phase1_arrays:
        return
    ref = np.asarray(phase1_arrays[key])
    if ref.shape != np.shape(ours) or not np.array_equal(ref, np.asarray(ours)):
        raise Phase1DependencyError(
            f"{model}/{tag}/{split}: the Phase-2 rows do not match the Phase-1 cell's cluster ids "
            f"({np.shape(ours)} vs {ref.shape}). The windows, the row filter or the cache changed "
            "since Phase 1 -- refusing to compare numbers across two different row sets.")


def chronos2_cache_candidates(tag: str, split: str, cache_dir=None) -> list[Path]:
    """The feature-cache files Phase 1 may have written for this split, in preference order.

    ``extraction._idf_prefix`` namespaces by the ACTIVE dataset set, which Phase 1's window
    builder mutates only when it BUILDS windows in-process (a window-cache hit leaves the default).
    So the same Phase-1 features may live under ``IDF_<tag>__<paper14 set>`` or ``IDF_<tag>``
    (or ``IDF_<tag>__ood`` for the cluster-split datasets). Every candidate is validated against
    the window labels before use; none is ever written.
    """
    from probing.config import CACHE_DIR
    from probing.id_data import OOD_TARGET_TAGS
    from probing.windows import PAPER14_SET
    base = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    K = math.ceil(H / 16)
    stem = f"K{K}_H{H}"
    if tag in OOD_TARGET_TAGS:
        prefixes = [f"IDF_{tag}__ood"]
    else:
        prefixes = [f"IDF_{tag}__{PAPER14_SET}", f"IDF_{tag}"]
    return [base / f"{p}__{split}__clean__{stem}.npz" for p in prefixes]


def _resolve_chronos2_cache(tag, split, Y, cache_dir=None) -> tuple[Path, list[str]]:
    tried = []
    for p in chronos2_cache_candidates(tag, split, cache_dir):
        if not p.exists():
            tried.append(f"{p.name}: absent")
            continue
        with np.load(p, allow_pickle=True) as z:
            y = z["y"]
        if len(y) == len(Y) and np.allclose(y, np.asarray(Y)):
            return p, tried
        tried.append(f"{p.name}: labels differ (another window set)")
    raise Phase1DependencyError(
        f"chronos2/{tag}/{split}: no Phase-1 forecast-slot cache matches these windows "
        f"({tried}). Phase 2 never extracts; re-run the Phase-1 cell for this dataset.")


def load_chronos2(tag: str, w: dict, phase1_arrays: dict, *, cache_dir=None,
                  splits=SPLITS) -> CellData:
    from probing.timesfm3 import raw_future_from_arcsinh
    spec = phase1.model_spec("chronos2")
    out, prov = {}, {"feature_files": {}}
    for split in splits:
        X = np.asarray(w[f"X_{split}"], np.float32)
        Y = np.asarray(w[f"Y_{split}_traj"], np.float32)
        path, tried = _resolve_chronos2_cache(tag, split, Y, cache_dir)
        with np.load(path, allow_pickle=True) as z:
            feats = {i: np.asarray(z[f"fslot_L{i}"], np.float32) for i in range(spec.n_depth_points)}
            head_in = np.asarray(z["fslot_final"], np.float32)
        for i, a in feats.items():
            if a.shape != (len(X), 4, spec.d):
                raise Phase1DependencyError(f"{path.name} L{i}: {a.shape}, expected "
                                            f"({len(X)}, 4, {spec.d})")
        X64 = X.astype(np.float64)
        rows = np.arange(len(X))
        cid = np.asarray(w[f"series_{split}"], np.int64)
        if split in ("val", "test"):
            check_cluster_ids("chronos2", tag, split, cid, phase1_arrays)
        out[split] = SplitData(
            feats=feats, target=Y, y_raw=raw_future_from_arcsinh(X, Y, w["meta"]["sigma_eps"]),
            X=X, inv={"mu": X64.mean(axis=1), "sd": np.maximum(X64.std(axis=1), 1e-6)},
            cluster_ids=cid, rows=rows, head_input=head_in)
        prov["feature_files"][split] = {"path": str(path), "candidates_tried": tried}
    return CellData("chronos2", tag, out, spec.depth_labels, spec.reference_index, prov)


def load_timesfm3(tag: str, w: dict, phase1_arrays: dict, p1cfg: dict, *, cache_dir,
                  splits=SPLITS) -> CellData:
    from probing.timesfm3 import raw_future_from_arcsinh
    from probing.timesfm3_last_token import (LastTokenGeometry, build_last_token_targets,
                                             cache_metadata, cache_root, read_cache)
    spec = phase1.model_spec("timesfm3")
    geom = LastTokenGeometry(C=C, H=H, strict=True)
    layers = list(range(spec.n_depth_points))
    detrend = bool(p1cfg.get("detrend", True))
    suite = p1cfg.get("suite", "paper14")
    out, prov = {}, {"cache_roots": {}}
    for split in splits:
        X = np.asarray(w[f"X_{split}"], np.float32)
        meta = cache_metadata(tag, split, geom, checkpoint=p1cfg["checkpoint"], detrend=detrend,
                              layers=layers, seed=int(p1cfg.get("seed", 0)),
                              feature_dtype=np.dtype(p1cfg.get("feature_dtype", "float32")),
                              suite=suite, n_windows=len(X))
        root = cache_root(cache_dir, tag, split, geom, detrend, suite)
        hit = read_cache(root, meta, X, layers)
        if hit is None:
            raise Phase1DependencyError(f"timesfm3/{tag}/{split}: no Phase-1 last-token cache at "
                                        f"{root}. Phase 2 never extracts.")
        Z = np.concatenate([X.astype(np.float64),
                            raw_future_from_arcsinh(X, w[f"Y_{split}_traj"],
                                                    w["meta"]["sigma_eps"])], axis=1)
        t = build_last_token_targets(Z, hit["mu"], hit["sd"], geom, detrend=detrend)
        rows = np.flatnonzero(t["valid"])
        cid = np.asarray(w[f"series_{split}"], np.int64)[rows]
        if split in ("val", "test"):
            check_cluster_ids("timesfm3", tag, split, cid, phase1_arrays)
        feats = {L: np.asarray(hit["feats"][L], np.float32)[rows][:, None, :] for L in layers}
        out[split] = SplitData(
            feats=feats, target=np.asarray(t["targets"], np.float32)[rows],
            y_raw=Z[rows, geom.target_start:geom.target_end], X=X[rows],
            inv={"mu": np.asarray(hit["mu"], np.float64)[rows],
                 "sd": np.asarray(hit["sd"], np.float64)[rows],
                 "trend": np.asarray(t["trend"], np.float64)[rows]},
            cluster_ids=cid, rows=rows,
            native_raw=np.asarray(hit["native"], np.float64)[rows])
        prov["cache_roots"][split] = str(root)
    prov["geometry"] = geom.as_dict()
    return CellData("timesfm3", tag, out, spec.depth_labels, spec.reference_index, prov)


def load_tirex(tag: str, w: dict, phase1_arrays: dict, p1cfg: dict, *, cache_dir, model, geom,
               splits=SPLITS) -> CellData:
    from probing.timesfm3 import raw_future_from_arcsinh
    from probing.tirex_model import (REP_NAMES, build_targets, cache_metadata, cache_root,
                                     read_cache)
    spec = phase1.model_spec("tirex")
    mode = p1cfg.get("rollout_mode", "two_pass")
    out, prov = {}, {"cache_files": {}}
    for split in splits:
        X = np.asarray(w[f"X_{split}"], np.float32)
        y_raw = raw_future_from_arcsinh(X, w[f"Y_{split}_traj"], w["meta"]["sigma_eps"])
        meta = cache_metadata(tag, split, geom, checkpoint=p1cfg["checkpoint"],
                              backend=p1cfg.get("backend", "torch"), points=list(REP_NAMES),
                              seed=int(p1cfg.get("seed", 0)), X=X, mode=mode,
                              batch_size=int(p1cfg.get("extract_batch_size", 256)))
        root = cache_root(cache_dir, tag, split, mode)
        got = read_cache(root, meta, list(REP_NAMES))
        if got is None:
            raise Phase1DependencyError(f"tirex/{tag}/{split}: no Phase-1 feature cache at "
                                        f"{root}.npz. Phase 2 never extracts.")
        named, native, _ = got
        tgt, loc, scale = build_targets(X, y_raw, model, geom, mode=mode)
        feats = {0: np.asarray(named["Emb"], np.float32)}
        for k in range(1, spec.num_blocks + 1):
            feats[k] = np.asarray(named[f"L{k}"], np.float32)
        rows = np.arange(len(X))
        cid = np.asarray(w[f"series_{split}"], np.int64)
        if split in ("val", "test"):
            check_cluster_ids("tirex", tag, split, cid, phase1_arrays)
        out[split] = SplitData(
            feats=feats, target=np.asarray(tgt, np.float32), y_raw=np.asarray(y_raw, np.float64),
            X=X, inv={"loc": np.asarray(loc, np.float64), "scale": np.asarray(scale, np.float64)},
            cluster_ids=cid, rows=rows, head_input=np.asarray(named[f"L{spec.num_blocks}+RMS"],
                                                              np.float32),
            native_raw=np.asarray(native, np.float64))
        prov["cache_files"][split] = str(root) + ".npz"
    prov["rollout_mode"] = mode
    return CellData("tirex", tag, out, spec.depth_labels, spec.reference_index, prov)
