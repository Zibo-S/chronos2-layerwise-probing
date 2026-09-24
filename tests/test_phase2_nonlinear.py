"""Contracts for the Phase 2b EXPLORATORY nested nonlinear adapter and its smoke driver.

Model-free: no checkpoint, no dataset, no cache. Login node, a few seconds, 2 threads:

    python -m tests.test_phase2_nonlinear

The committed H3 path must be untouched: N3 pins that the default ``fit_residual_adapter`` (no
``make_adapter``) is bitwise the explicit ``ResidualAdapter`` factory, and the existing
``tests.test_phase2_h3`` suite must still pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.phase2_align import (NestedResidualAdapter, ResidualAdapter,       # noqa: E402
                                  fit_residual_adapter, mse_loss)


def _frozen_head(d_in, d_out, seed=0):
    torch.manual_seed(seed)
    head = nn.Linear(d_in, d_out)
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def test_n1_nested_is_bitwise_affine_and_hard_cut_at_init():
    torch.manual_seed(0)
    x = torch.randn(7, 3, 16) * 50.0                       # unnormalized, like TimesFM-3's stream
    nested = NestedResidualAdapter(16, r=4)
    assert torch.equal(nested(x), x), "nested adapter at init is not bitwise the hard cut"
    affine = ResidualAdapter(16)
    with torch.no_grad():
        for m in (nested, affine):
            m.delta.copy_(torch.linspace(-0.1, 0.1, 256).view(16, 16))
            m.bias.copy_(torch.linspace(-1, 1, 16))
    assert torch.equal(nested(x), affine(x)), "with W2 = 0 the nested adapter is not the affine one"
    print(" N1  nested adapter: W2=0 => bitwise the affine adapter, and the hard cut at init  OK")


def test_n2_parameter_count_and_decay_groups():
    d, r = 32, 8
    a = NestedResidualAdapter(d, r=r)
    assert a.n_params == d * d + d + r * d + r + d * r
    assert sum(p.numel() for p in a.parameters()) == a.n_params
    dec = {id(p) for p in a.decay_params()}
    free = {id(p) for p in a.free_params()}
    assert dec == {id(a.delta), id(a.w1), id(a.w2)} and free == {id(a.bias), id(a.b1)}
    assert dec | free == {id(p) for p in a.parameters()} and not dec & free
    b = ResidualAdapter(d)
    assert [id(p) for p in b.decay_params()] == [id(b.delta)]
    assert [id(p) for p in b.free_params()] == [id(b.bias)]
    print(" N2  parameter count d^2+d+2rd+r; every parameter in exactly one decay group  OK")


def _toy(nonlinear: bool, seed=0):
    head = _frozen_head(8, 6, seed)
    g = torch.Generator().manual_seed(seed + 1)
    x_tr, x_va = torch.randn(200, 8, generator=g), torch.randn(100, 8, generator=g)
    U = torch.randn(8, 8, generator=g)

    def target(x):
        return head(x + (0.8 * torch.tanh(x @ U.T) if nonlinear else x @ U.T * 0.3))
    return dict(d=8, pathway_fn=head, train_x=x_tr, train_y=target(x_tr), val_x=x_va,
                val_y=target(x_va), train_loss=mse_loss, val_criterion=mse_loss,
                wd_grid=(0.0, 1e-2), epochs=200, lr=1e-2, eval_every=10, device="cpu")


def test_n3_default_path_is_bitwise_the_explicit_affine_factory():
    kw = _toy(nonlinear=False)
    r_default = fit_residual_adapter(**kw)
    r_explicit = fit_residual_adapter(**kw, make_adapter=lambda d: ResidualAdapter(d))
    assert torch.equal(r_default["adapter"].delta, r_explicit["adapter"].delta)
    assert torch.equal(r_default["adapter"].bias, r_explicit["adapter"].bias)
    assert r_default["selected_epoch"] == r_explicit["selected_epoch"]
    assert r_default["adapter_class"] == "ResidualAdapter"
    print(" N3  fit_residual_adapter default == explicit ResidualAdapter factory (bitwise)  OK")


def test_n4_nested_fit_nests_hard_cut_is_deterministic_and_helps_a_nonlinear_target():
    kw = _toy(nonlinear=True)
    nest = lambda d: NestedResidualAdapter(d, r=16)                       # noqa: E731
    a = fit_residual_adapter(**kw)
    n1 = fit_residual_adapter(**kw, make_adapter=nest)
    n2 = fit_residual_adapter(**kw, make_adapter=nest)
    assert n1["hard_cut_val"] == a["hard_cut_val"], "epoch 0 of the nested fit is not the hard cut"
    assert torch.equal(n1["adapter"].w2, n2["adapter"].w2), "nested fits are not deterministic"
    assert n1["adapter_class"] == "NestedResidualAdapter"
    assert n1["selected_val"] < a["selected_val"], \
        f"nested {n1['selected_val']:.4g} did not beat affine {a['selected_val']:.4g} on a " \
        "nonlinear target"
    with torch.no_grad():                                  # the returned module IS the selection
        v = float(mse_loss(kw["pathway_fn"](n1["adapter"](kw["val_x"])), kw["val_y"]))
    assert abs(v - n1["selected_val"]) <= 1e-6 * max(1.0, abs(v))
    print(f" N4  nested fit: epoch 0 = hard cut; deterministic; nonlinear toy val "
          f"{n1['selected_val']:.4f} < affine {a['selected_val']:.4f}  OK")


def test_n5_driver_depths_r2_and_decision_rule():
    from experiments.run_phase2_nonlinear_smoke import RULE, decide, r2, representative_depths
    assert representative_depths(12) == [3, 6, 9] and representative_depths(20) == [5, 10, 15]
    y = np.random.default_rng(0).normal(size=(50, 4))
    assert r2(y, y) == 1.0 and abs(r2(np.broadcast_to(y.mean(0), y.shape), y)) < 1e-12

    def rows(t_ref, t_new, other_aff, other_nest):
        out = []
        for i, d in enumerate((5, 10, 15)):
            out += [{"model": "timesfm3", "arm": "affine_closed", "depth_index": d,
                     "val_mase_ratio": t_ref[i]},
                    {"model": "timesfm3", "arm": "nested", "depth_index": d,
                     "val_mase_ratio": t_new[i]}]
        for m in ("chronos2", "tirex"):
            for i, d in enumerate((3, 6, 9)):
                out += [{"model": m, "arm": "affine_iter", "depth_index": d,
                         "val_mase_ratio": other_aff[i]},
                        {"model": m, "arm": "nested", "depth_index": d,
                         "val_mase_ratio": other_nest[i]}]
        return out
    ok = decide(rows([1.7, 1.7, 1.5], [1.4, 1.5, 1.45], [1.05] * 3, [1.05] * 3))
    assert ok["adopt"] is True and ok["timesfm3_n_meaningful"] == 2
    small = decide(rows([1.7, 1.7, 1.5], [1.65, 1.66, 1.49], [1.05] * 3, [1.05] * 3))
    assert small["adopt"] is False, "gains < 10 pp must not adopt"
    worse = decide(rows([1.7, 1.7, 1.5], [1.4, 1.5, 1.45], [1.05] * 3, [1.07] * 3))
    assert worse["adopt"] is False, "a > 1 pp loss on Chronos-2 / TiRex must not adopt"
    one_worse = decide(rows([1.7, 1.7, 1.5], [1.4, 1.5, 1.55], [1.05] * 3, [1.05] * 3))
    assert one_worse["adopt"] is False, "must improve at ALL three depths"
    partial = decide([r for r in rows([1.7] * 3, [1.4] * 3, [1.0] * 3, [1.0] * 3)
                      if r["model"] != "tirex"])
    assert partial["adopt"] is None and partial["complete"] is False
    assert RULE["min_gain_pp"] == 10.0 and RULE["max_other_loss_pp"] == 1.0
    print(" N5  driver: depths 25/50/75%, R^2, and the pre-registered rule (5 cases)  OK")


def test_n6_adapter_persistence_nested_round_trip_and_affine_checksum_unchanged():
    import shutil
    import tempfile
    from probing.phase2 import array_sha256
    from probing.phase2_align import adapter_arrays_sha256, load_adapter, save_adapter
    torch.manual_seed(0)
    aff = ResidualAdapter(16)
    nes = NestedResidualAdapter(16, r=4)
    with torch.no_grad():
        for m in (aff, nes):
            m.delta.normal_()
            m.bias.normal_()
        nes.w2.normal_()
    # the affine checksum is EXACTLY the pre-2b formula, so committed adapters still verify
    assert adapter_arrays_sha256(aff) == array_sha256(aff.delta.detach().cpu().numpy(),
                                                      aff.bias.detach().cpu().numpy())
    tmp = Path(tempfile.mkdtemp())
    try:
        x = torch.randn(5, 16)
        for ad in (aff, nes):
            rec = save_adapter(tmp / f"{type(ad).__name__}.npz", ad, {"k": 1})
            back, meta = load_adapter(rec["path"])
            assert type(back) is type(ad) and meta["sha256"] == adapter_arrays_sha256(ad)
            assert torch.equal(back(x), ad.eval()(x)), "round trip is not bitwise"
            assert rec["n_params"] == ad.n_params
        # a tampered nested branch must fail the checksum
        with np.load(tmp / "NestedResidualAdapter.npz") as z:
            arrs = {k: z[k] for k in z.files}
        arrs["w2"] = arrs["w2"] + 1.0
        np.savez(tmp / "bad.npz", **arrs)
        try:
            load_adapter(tmp / "bad.npz")
            raise AssertionError("a tampered nested adapter loaded")
        except RuntimeError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(" N6  adapter files: nested round trip bitwise + checksummed; affine checksum unchanged OK")


def test_n7_whole_h3_cell_with_the_nested_noa_arm():
    from probing.phase2_h3 import run_h3_cell
    from tests.test_phase2_h3 import FAST, _mock_cell
    pw, data, arrays, ent = _mock_cell()
    res = run_h3_cell(pw, data, arrays, entrances=ent, families=["hard", "noa"], device="cpu",
                      fit_cfg={**FAST, "noa_adapter": "nested", "bottleneck": 4},
                      adapter_dir=None, floor_val_loss=1.0, save_prediction_depths=[6],
                      log=lambda *a: None)
    row = {(r["family"], r["label"]): r for r in res["rows"]}
    assert row[("noa", "L12")]["mase_ratio_vs_native"] == 1.0
    assert row[("noa", "L11")]["test_loss"] <= row[("hard", "L11")]["test_loss"] * 1.001
    fits = res["fits"]["noa"]
    fitted = [f for f in fits.values() if f.get("fitted")]
    assert fitted and all(f.get("adapter_class") == "NestedResidualAdapter" for f in fitted)
    try:
        run_h3_cell(pw, data, arrays, entrances=ent, families=["hard"], device="cpu",
                    fit_cfg={**FAST, "noa_adapter": "mlp"}, log=lambda *a: None)
        raise AssertionError("an unknown noa_adapter was accepted")
    except ValueError:
        pass
    print(" N7  whole H3 cell with noa_adapter=nested: every fitted NOA is nested; final = native OK")


def test_n8_nested_option_changes_the_cell_hash_only_when_set():
    from experiments.run_phase2_h3 import cell_config, parse_args
    dep = {"fit_hash": "f", "config_hash": "c", "window_digest": "w"}
    a = parse_args([])
    n = parse_args(["--noa-adapter", "nested"])
    assert a.output_root.endswith("three_model_phase2") and n.output_root.endswith("_nested")
    assert "adapters_nested" in n.adapter_root and "adapters_nested" not in a.adapter_root
    base = {"epochs": 1, "lr": 1e-3, "wd_grid": [0.0], "eval_every": 1, "kappas": [1.0],
            "seed": 0, "native_gate_rtol": 1e-4, "boot_b": 10, "boot_seed": 0}
    c_aff = cell_config("tirex", "t", dep, ["hard"], None, {**base, "noa_adapter": "affine"}, a)
    c_old = cell_config("tirex", "t", dep, ["hard"], None, base, a)
    c_nes = cell_config("tirex", "t", dep, ["hard"], None,
                        {**base, "noa_adapter": "nested", "bottleneck": 64}, n)
    assert c_aff == c_old, "the affine option changed the committed cell config"
    assert c_nes["noa_adapter"] == "nested" and c_nes != c_aff
    print(" N8  cell hash: affine configs unchanged; nested gets its own hash and output tree  OK")


def main(argv=None):
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"\nPHASE-2b NONLINEAR CONTRACTS  ({len(tests)} model-free groups)\n" + "=" * 78)
    failed = []
    for t in tests:
        try:
            t()
        except Exception:
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print("=" * 78)
    if failed:
        print(f"{len(failed)} of {len(tests)} contracts FAILED: {', '.join(failed)}")
        return 1
    print(f"all {len(tests)} contracts hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
