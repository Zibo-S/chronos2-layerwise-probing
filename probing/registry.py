"""THE central dataset registry — one source of truth for every dataset fact.

Before this module the same facts were re-declared in a dozen places: the roster in
``probing/tunnel.py`` AND ``experiments/run_cka_analysis.py`` AND
``experiments/run_compression_cost.py`` AND ``experiments/make_erank_stability_figure.py``;
display names in ``make_id_paper_figures.py`` AND ``run_timesfm3_last_token_probing.py``;
the seasonal period as a single global ``M_SEASON = 24``. Every duplicate was a chance for
two drivers to disagree about the same dataset, and the global ``m = 24`` silently produced a
two-hour "season" for 5-minute data.

Everything a driver needs to know about a dataset now lives in exactly one row here, and every
lookup FAILS LOUDLY on an unknown tag. There is no default seasonal period, no fallback
display name and no inferred builder.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
  * it does not import numpy, torch, ``datasets`` or any project module — it is pure data plus
    lookups, so it can be imported from anywhere (including ``probing.id_data``, which the
    experiment drivers import) without a cycle;
  * it does not define the tunnel criterion, which stays in ``probing/tunnel.py`` unchanged;
  * it does not decide that an unlisted dataset is "OOD". See ``Provenance``.

THE ROSTER. ``paper14`` is the benchmark frozen at the 2026-09-21 yield-screen gate
(``notes/PLAN.md``). ``paper7`` is the original committed seven, kept as an exact subset so the
Chronos-2 / TimesFM-3 / TiRex numbers already in ``results/`` stay reproducible. ``legacy``
carries the older dataset-set tags (phase0_trio / extended_v1 / extended_v2) that committed
Chronos-2 drivers still load; they are registry members so ``seasonal_m`` resolves for them
instead of raising.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "DatasetSpec", "Provenance", "DATASETS", "PROVENANCE", "MODELS",
    "PAPER14", "PAPER7", "STATUSES", "EVIDENCE_SCOPES", "BUILDERS", "ROLES",
    "spec", "seasonal_m", "display_name", "builder", "cluster_unit", "max_series",
    "role", "is_control", "roster", "tags", "known", "provenance", "slug",
    "unverified_provenance", "primary_datasets", "control_datasets",
    "MASE_DEFINITION", "SERIES_CAP_RNG_STREAM",
]


# --------------------------------------------------------------------------- #
# the reported MASE definition — ONE string, quoted by every driver and the paper
# --------------------------------------------------------------------------- #
# The reported metric has ALWAYS been the in-context denominator (run_id_forecasting.
# _mase_denominator / run_timesfm3_probing.mase_denominator); id_data's canonical
# `test_denominator` is computed but consumed by nothing. This refactor PRESERVES the reported
# definition exactly and only makes `m` per-dataset instead of a global 24.
MASE_DEFINITION = (
    "MASE is normalized by the seasonal-naive error computed over the 512-step context "
    "window using the dataset-specific seasonal period."
)

# Dedicated RNG stream id for the deterministic series cap. It is spawned as
# `np.random.default_rng([seed, SERIES_CAP_RNG_STREAM])` so the cap draws from a stream that is
# INDEPENDENT of the builder's main `rng` — a dataset small enough not to be capped therefore
# consumes zero random numbers here and its windows stay byte-identical.
SERIES_CAP_RNG_STREAM = 0x5E12E5


# --------------------------------------------------------------------------- #
# provenance: status x evidence_scope, never collapsed to one axis
# --------------------------------------------------------------------------- #
STATUSES = ("pretraining_exposed", "explicitly_held_out", "not_listed", "undocumented")
EVIDENCE_SCOPES = ("exact_dataset", "benchmark_exclusion", "source_level", "source_family")
MODELS = ("chronos2", "timesfm3", "tirex")


@dataclass(frozen=True)
class Provenance:
    """One (dataset, model) pretraining-provenance cell.

    The two axes are kept separate ON PURPOSE. ``status`` alone would let a dataset whose only
    evidence is "the model card names Wikipedia Pageviews" read as identical evidence to one
    the paper lists by name in its corpus table. It is not, and any count in a paper table must
    be reported per (status, evidence_scope) cell.

    ``not_listed`` means EXACTLY "absent from the documentation we have". It is NOT a synonym
    for OOD, unseen or held-out; only ``explicitly_held_out`` carries an author's own exclusion
    statement. Nothing in this codebase may convert one into the other.

    ``verified`` records whether the citation was checked against the primary source by a human.
    An unverified cell is usable as a working assumption but must not be cited in the paper
    until checked — ``unverified_provenance()`` lists them.
    """
    status: str
    evidence_scope: str
    citation: str
    verified: bool = False

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status {self.status!r} not in {STATUSES}")
        if self.evidence_scope not in EVIDENCE_SCOPES:
            raise ValueError(f"evidence_scope {self.evidence_scope!r} not in {EVIDENCE_SCOPES}")
        if not self.citation.strip():
            raise ValueError("every provenance cell needs citation/evidence text")

    def as_dict(self) -> dict:
        return {"status": self.status, "evidence_scope": self.evidence_scope,
                "citation": self.citation, "verified": bool(self.verified)}


# --------------------------------------------------------------------------- #
# the dataset row
# --------------------------------------------------------------------------- #
# Builder names describe the WINDOW PROTOCOL, deliberately not a pretraining status. They were
# briefly "pt_id_rolling" / "ood_rolling", which re-encoded the Chronos-2-relative taxonomy this
# refactor exists to remove -- and made a dataset row literally contain the string "pt_id".
#   rolling_within_series : id_data.build_windows under a ROLLING_SETS dataset set; one series
#                           per bootstrap unit, H-spaced origins, last -> test, 2nd-last -> val.
#   rolling_cluster       : id_data.build_ood_rolling_windows; the bootstrap unit is the PARENT
#                           CLUSTER (carpark / station / metric-query), several series per unit.
#   legacy_auto_split     : the pre-rolling length-derived within_series/cross_series path.
BUILDERS = ("rolling_within_series", "rolling_cluster", "legacy_auto_split")
ROLES = ("primary", "control")


@dataclass(frozen=True)
class DatasetSpec:
    """Everything a driver may need to know about one dataset.

    ``builder`` decides which window constructor runs — this is what replaces
    ``tag in PT_OOD_TAGS`` as the dataset-validity switch. A dataset's window protocol is a
    property of the dataset, not of some model's pretraining corpus.

    ``max_series`` is the DETERMINISTIC series-level cap applied before the rolling protocol
    (``id_data._build_rolling_windows``). ``None`` means "no cap"; the cap is a no-op — and
    consumes no randomness — whenever the eligible pool is already at or below it, which is why
    the original seven are provably unaffected.

    ``role`` is ``"control"`` for a dataset that exists to hold something fixed rather than to
    add domain diversity. A control may never be counted toward domain coverage.
    """
    tag: str
    display_name: str
    freq: str
    seasonal_m: int
    domain: str
    cluster_unit: str
    builder: str
    roster: str                       # "paper14" | "alternate" | "legacy"
    slug: str = ""                    # filesystem-safe short name; defaults to a slugified tag
    role: str = "primary"
    source_repo: str | None = None    # HF repo id; None for staged arrow shards
    source_config: str | None = None
    target_column: str | None = None
    max_series: int | None = None
    notes: str = ""

    def __post_init__(self):
        if not self.slug:
            object.__setattr__(self, "slug", _slugify(self.tag))
        if self.seasonal_m < 1:
            raise ValueError(f"{self.tag}: seasonal_m must be >= 1, got {self.seasonal_m}")
        if self.builder not in BUILDERS:
            raise ValueError(f"{self.tag}: builder {self.builder!r} not in {BUILDERS}")
        if self.role not in ROLES:
            raise ValueError(f"{self.tag}: role {self.role!r} not in {ROLES}")
        if self.builder != "rolling_cluster" and not (self.source_repo and self.source_config
                                                  and self.target_column):
            raise ValueError(f"{self.tag}: an HF-sourced dataset needs repo/config/target column")
        if self.max_series is not None and self.max_series < 1:
            raise ValueError(f"{self.tag}: max_series must be positive or None")

    def as_dict(self) -> dict:
        return {"tag": self.tag, "display_name": self.display_name, "slug": self.slug,
                "freq": self.freq,
                "seasonal_m": self.seasonal_m, "domain": self.domain,
                "cluster_unit": self.cluster_unit, "builder": self.builder,
                "roster": self.roster, "role": self.role, "source_repo": self.source_repo,
                "source_config": self.source_config, "target_column": self.target_column,
                "max_series": self.max_series, "notes": self.notes}


def _slugify(tag: str) -> str:
    """Lowercase, filesystem-safe form of a tag. Only used when a row declares no slug."""
    return re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")


_CHRONOS = "autogluon/chronos_datasets"
_FEV = "autogluon/fev_datasets"

# The deterministic series cap for the huge rosters. It equals the rolling train budget
# (BUDGET_BY_SET["extended_v3_rolling"][0] = 1394): capping the ELIGIBLE pool at the train
# budget is exactly the condition `_build_rolling_windows`' fail-loud coverage check needs, so
# every retained series keeps >= 1 train window and the 1394/262/262 budget is met exactly.
_TRAIN_BUDGET_CAP = 1394


def _ds(**kw) -> DatasetSpec:
    return DatasetSpec(**kw)


# --------------------------------------------------------------------------- #
# THE ROSTER
# --------------------------------------------------------------------------- #
_ROWS: tuple[DatasetSpec, ...] = (
    # ---------------- paper14, the original seven (committed results exist) ----------------
    _ds(tag="m4_hourly", slug="m4", display_name="M4", freq="1H", seasonal_m=24,
        domain="mixed hourly", cluster_unit="series", builder="rolling_within_series",
        roster="paper14", source_repo=_CHRONOS, source_config="m4_hourly",
        target_column="target"),
    _ds(tag="monash_electricity_hourly", slug="electricity", display_name="Electricity", freq="1H", seasonal_m=24,
        domain="energy", cluster_unit="series", builder="rolling_within_series",
        roster="paper14", source_repo=_CHRONOS, source_config="monash_electricity_hourly",
        target_column="target"),
    _ds(tag="uber_tlc_hourly", slug="uber_tlc", display_name="Uber TLC", freq="1H", seasonal_m=24,
        domain="transport / ride demand", cluster_unit="series", builder="rolling_within_series",
        roster="paper14", source_repo=_CHRONOS, source_config="uber_tlc_hourly",
        target_column="target"),
    _ds(tag="wind_farms_hourly", slug="wind_farms", display_name="Wind Farms", freq="1H", seasonal_m=24,
        domain="renewable generation", cluster_unit="series", builder="rolling_within_series",
        roster="paper14", source_repo=_CHRONOS, source_config="wind_farms_hourly",
        target_column="target",
        notes="yield screen 2026-09-21: 21.95% of val/test windows have a CONSTANT future "
              "(mad_const == 0), the highest in the roster; those windows score ~0 at every "
              "depth and compress the layer-to-layer differences."),
    _ds(tag="sg_carpark", display_name="SG Carpark", freq="1H(agg)", seasonal_m=24,
        domain="transport / parking", cluster_unit="carpark", builder="rolling_cluster",
        roster="paper14"),
    _ds(tag="coastal_ts", display_name="Coastal T-S", freq="1H", seasonal_m=24,
        domain="nature / oceanography", cluster_unit="station", builder="rolling_cluster",
        roster="paper14",
        notes="24 clusters (2 variates per station) -> 48 val/test windows; the weakest "
              "statistical support in the roster."),
    _ds(tag="boom_hourly", slug="boom", display_name="BOOM", freq="1H", seasonal_m=24,
        domain="cloud observability", cluster_unit="metric_query", builder="rolling_cluster",
        roster="paper14",
        notes="yield screen: median constant-forecast MAD 0.121, ~3x lower than any other "
              "dataset -> least dynamic range; small absolute deltas mean less here."),

    # ---------------- paper14, the seven additions ----------------
    _ds(tag="LOOP_SEATTLE_5T", display_name="Loop Seattle", freq="5min", seasonal_m=288,
        domain="road speed", cluster_unit="series", builder="rolling_within_series",
        roster="paper14", source_repo=_FEV, source_config="LOOP_SEATTLE_5T",
        target_column="target"),
    _ds(tag="electricity_15min", display_name="Electricity 15min", freq="15min", seasonal_m=96,
        domain="energy", cluster_unit="series", builder="rolling_within_series",
        roster="paper14", role="control", source_repo=_CHRONOS,
        source_config="electricity_15min", target_column="consumption_kW",
        notes="FREQUENCY CONTROL, not domain diversity: the same 370 meters as "
              "monash_electricity_hourly at 4x the sampling rate. It holds domain and series "
              "fixed and varies only the rate. Never count it toward domain coverage."),
    _ds(tag="SZ_TAXI_15T", display_name="SZ Taxi", freq="15min", seasonal_m=96,
        domain="road speed", cluster_unit="series", builder="rolling_within_series",
        roster="paper14", source_repo=_FEV, source_config="SZ_TAXI_15T",
        target_column="target"),
    _ds(tag="monash_london_smart_meters", display_name="London Smart Meters", freq="30min",
        seasonal_m=48, domain="residential energy", cluster_unit="series",
        builder="rolling_within_series", roster="paper14", source_repo=_CHRONOS,
        source_config="monash_london_smart_meters", target_column="target",
        max_series=_TRAIN_BUDGET_CAP,
        notes="5555 eligible series -> deterministic cap. Sparse isolated NaNs poison the "
              "unused canonical denominator for 98.5% of series; the REPORTED in-context "
              "denominator is unaffected (the context is finite by construction)."),
    _ds(tag="m5", display_name="M5", freq="1D", seasonal_m=7,
        domain="retail", cluster_unit="series", builder="rolling_within_series",
        roster="paper14", source_repo=_CHRONOS, source_config="m5", target_column="target",
        max_series=_TRAIN_BUDGET_CAP,
        notes="28491 eligible series -> deterministic cap. The yield screen's degeneracy "
              "trigger did NOT fire (median constant-forecast MAD 0.490), so rossmann_1D "
              "stays a benched alternate."),
    _ds(tag="wiki_daily_100k", display_name="Wiki Daily", freq="1D", seasonal_m=7,
        domain="web traffic", cluster_unit="series", builder="rolling_within_series",
        roster="paper14", source_repo=_CHRONOS, source_config="wiki_daily_100k",
        target_column="target", max_series=_TRAIN_BUDGET_CAP,
        notes="100000 eligible series -> deterministic cap; the largest raw load in the "
              "roster (~2.2 GB of float64 series)."),
    _ds(tag="monash_traffic", display_name="Traffic", freq="1H", seasonal_m=24,
        domain="road volume", cluster_unit="series", builder="rolling_within_series",
        roster="paper14", source_repo=_CHRONOS, source_config="monash_traffic",
        target_column="target"),

    # ---------------- designated alternates: screened, NOT in the roster ----------------
    _ds(tag="rossmann_1D", display_name="Rossmann", freq="1D", seasonal_m=7,
        domain="retail", cluster_unit="series", builder="rolling_within_series",
        roster="alternate", source_repo=_FEV, source_config="rossmann_1D",
        target_column="Sales",
        notes="M5 fallback; benched because M5's degeneracy trigger did not fire."),
    _ds(tag="kdd_cup_2022_10T", display_name="KDD Cup 2022", freq="10min", seasonal_m=144,
        domain="wind power", cluster_unit="series", builder="rolling_within_series",
        roster="alternate", source_repo=_FEV, source_config="kdd_cup_2022_10T",
        target_column="Patv",
        notes="DROPPED at the 2026-09-21 roster gate: 75.2% of candidate windows rejected "
              "(the evaluation would land on a missingness-selected 25% of time) and val/test "
              "caps at 134 clusters. Its wind-power domain is covered by wind_farms_hourly. "
              "NOTE: this config has no 'target' column — fev's own task declares Patv."),

    # ---------------- legacy tags: older dataset sets committed drivers still load ---------
    _ds(tag="solar_1h", display_name="Solar", freq="1H", seasonal_m=24,
        domain="solar generation", cluster_unit="series", builder="legacy_auto_split",
        roster="legacy", source_repo=_CHRONOS, source_config="solar_1h",
        target_column="power_mw",
        notes="phase0_trio only; carries the documented label pathology (paper/phase0_fixes.md)."),
    _ds(tag="monash_kdd_cup_2018", display_name="KDD Cup 2018", freq="1H", seasonal_m=24,
        domain="air quality", cluster_unit="series", builder="legacy_auto_split",
        roster="legacy", source_repo=_CHRONOS, source_config="monash_kdd_cup_2018",
        target_column="target", notes="extended_v1 only."),
    _ds(tag="monash_pedestrian_counts", display_name="Pedestrian Counts", freq="1H",
        seasonal_m=24, domain="foot traffic", cluster_unit="series",
        builder="legacy_auto_split", roster="legacy", source_repo=_CHRONOS,
        source_config="monash_pedestrian_counts", target_column="target",
        notes="extended_v1 only."),
)

DATASETS: dict[str, DatasetSpec] = {}
for _row in _ROWS:
    if _row.tag in DATASETS:
        raise RuntimeError(f"duplicate registry tag {_row.tag!r}")
    DATASETS[_row.tag] = _row
del _row

#: The frozen 14-dataset benchmark, in the order the paper's tables and figures use.
PAPER14: tuple[str, ...] = tuple(t for t, s in DATASETS.items() if s.roster == "paper14")

#: The original committed seven, an EXACT subset of PAPER14 and the order the committed
#: Chronos-2 / TimesFM-3 / TiRex artifacts use (four rolling_within_series first, then three
#: ood_rolling). Kept so existing results stay reproducible.
PAPER7: tuple[str, ...] = ("monash_electricity_hourly", "uber_tlc_hourly", "m4_hourly",
                           "wind_farms_hourly", "sg_carpark", "coastal_ts", "boom_hourly")


# --------------------------------------------------------------------------- #
# provenance table: one cell per (dataset, model)
# --------------------------------------------------------------------------- #
_T6 = ("Ansari et al., Chronos-2 (arXiv:2510.15821), Table 6 — the full list of real "
       "univariate pretraining datasets")
_FEVBENCH = ("Chronos-2 (arXiv:2510.15821) Sec 5.1 on fev-bench: \"None of these datasets or "
             "tasks were seen by Chronos-2 during training.\"")
_BENCH2 = ("Chronos-2 (arXiv:2510.15821) on Chronos Benchmark II: \"None of these datasets "
           "were included in the training corpus of Chronos-2.\"")
_TFM3_CARD = ("TimesFM-3 model card (google/timesfm-3.0-pytorch; blog 2026-08-31, no paper): "
              "corpus = GiftEvalPretrain excluding datasets overlapping fev-bench, + Wikipedia "
              "Pageviews (cutoff Nov 2023), + Google Trends top queries (cutoff EoY 2022), + "
              "\"synthetic and augmented data\" (undefined)")
_TIREX_C = ("TiRex (arXiv:2505.23719) App C.2/C.3: Chronos-1 corpus (Table 5) + a GIFT-Eval "
            "subset (Table 6) + 15M synthetic GP series; 16 of 97 GIFT-Eval settings are "
            "excluded but ARE NEVER NAMED, so per-dataset status cannot be derived")
_TIREX_ZS = ("TiRex (arXiv:2505.23719): \"TiRex's pre-training data has no overlap with "
             "Chronos-ZS benchmark.\"")

_UNDOC_TFM3 = Provenance("undocumented", "source_family", _TFM3_CARD, verified=True)
_UNDOC_TIREX = Provenance("undocumented", "source_family", _TIREX_C, verified=True)

PROVENANCE: dict[tuple[str, str], Provenance] = {
    # ---------------- Chronos-2 ----------------
    ("m4_hourly", "chronos2"): Provenance(
        "pretraining_exposed", "exact_dataset", _T6 + " lists \"M4 (Hourly/Daily/Weekly/"
        "Monthly)\".", verified=True),
    ("monash_electricity_hourly", "chronos2"): Provenance(
        "pretraining_exposed", "exact_dataset", _T6 + " lists \"Electricity\".", verified=True),
    ("uber_tlc_hourly", "chronos2"): Provenance(
        "pretraining_exposed", "source_family", _T6 + " lists \"Taxi\" (NYC TLC trip records). "
        "Uber TLC is the same NYC TLC source family, not a verbatim Table 6 entry — this is "
        "family-level evidence, deliberately NOT recorded as exact_dataset.", verified=True),
    ("wind_farms_hourly", "chronos2"): Provenance(
        "pretraining_exposed", "exact_dataset", _T6 + " lists \"Wind Farms\".", verified=True),
    ("sg_carpark", "chronos2"): Provenance(
        "not_listed", "exact_dataset", "Absent from " + _T6 + ", which the paper declares "
        "exhaustive for real univariate data. No cutoff date is stated anywhere in the paper, "
        "so absence is not a temporal guarantee.", verified=True),
    ("coastal_ts", "chronos2"): Provenance(
        "not_listed", "exact_dataset", "Absent from " + _T6 + " (declared exhaustive). No "
        "cutoff date is stated.", verified=True),
    ("boom_hourly", "chronos2"): Provenance(
        "not_listed", "benchmark_exclusion",
        "CORRECTED 2026-09-21: the strings \"BOOM\" and \"Datadog\" do NOT appear in "
        "arXiv:2510.15821, so data/chronos2_seen_manifest.md's \"explicitly listed\" claim is "
        "wrong. Held-out status is an INFERENCE via fev-bench, which contains BOOMLET (a BOOM "
        "subset). " + _FEVBENCH, verified=True),
    ("LOOP_SEATTLE_5T", "chronos2"): Provenance(
        "explicitly_held_out", "benchmark_exclusion", _FEVBENCH +
        " VERIFY: confirm LOOP_SEATTLE is a fev-bench task and not merely a member of the "
        "autogluon/fev_datasets collection."),
    ("SZ_TAXI_15T", "chronos2"): Provenance(
        "explicitly_held_out", "benchmark_exclusion", _FEVBENCH +
        " VERIFY: confirm SZ_TAXI is a fev-bench task and not merely a member of the "
        "autogluon/fev_datasets collection."),
    ("electricity_15min", "chronos2"): Provenance(
        "pretraining_exposed", "source_level", _T6 + " lists \"Electricity\". The 15-minute "
        "variant is the SAME 370 meters at 4x the rate, so exposure is source-level: the "
        "underlying measurements were seen, this exact resampling was not listed.",
        verified=True),
    ("monash_london_smart_meters", "chronos2"): Provenance(
        "pretraining_exposed", "exact_dataset", _T6 + " lists \"London Smart Meters\".",
        verified=True),
    ("m5", "chronos2"): Provenance(
        "explicitly_held_out", "benchmark_exclusion", _BENCH2 +
        " VERIFY: confirm M5 is a Chronos Benchmark II task."),
    ("wiki_daily_100k", "chronos2"): Provenance(
        "pretraining_exposed", "exact_dataset", _T6 + " lists \"Wiki\".", verified=True),
    ("monash_traffic", "chronos2"): Provenance(
        "undocumented", "source_family", _T6 + " is declared exhaustive, but the transcription "
        "in data/chronos2_seen_manifest.md ends in \"a.o.\" and does not settle whether "
        "\"Traffic\" is an entry. VERIFY against the published Table 6 before citing."),
    ("rossmann_1D", "chronos2"): Provenance(
        "undocumented", "source_family", _T6 + " — not transcribed for this benched alternate."),
    ("kdd_cup_2022_10T", "chronos2"): Provenance(
        "undocumented", "source_family", _T6 + " — not transcribed for this dropped dataset."),
    ("solar_1h", "chronos2"): Provenance(
        "pretraining_exposed", "exact_dataset", _T6 + " lists \"Solar\".", verified=True),
    ("monash_kdd_cup_2018", "chronos2"): Provenance(
        "pretraining_exposed", "exact_dataset", _T6 + " lists \"KDD Cup 2018\".", verified=True),
    ("monash_pedestrian_counts", "chronos2"): Provenance(
        "undocumented", "source_family", _T6 + " — legacy extended_v1 tag, not transcribed."),

    # ---------------- TimesFM-3 ----------------
    ("wiki_daily_100k", "timesfm3"): Provenance(
        "pretraining_exposed", "source_level", _TFM3_CARD + ". The card names the SOURCE "
        "(\"Wikipedia Pageviews, cutoff Nov 2023\"); our series end 2022-12-31, inside that "
        "cutoff. This is source-level evidence — the card never names wiki_daily_100k."),
    ("m5", "timesfm3"): Provenance(
        "explicitly_held_out", "benchmark_exclusion", _TFM3_CARD +
        ". The corpus excludes datasets overlapping fev-bench. VERIFY M5's fev-bench membership."),
    ("LOOP_SEATTLE_5T", "timesfm3"): Provenance(
        "explicitly_held_out", "benchmark_exclusion", _TFM3_CARD + " VERIFY fev-bench membership."),
    ("SZ_TAXI_15T", "timesfm3"): Provenance(
        "explicitly_held_out", "benchmark_exclusion", _TFM3_CARD + " VERIFY fev-bench membership."),

    # ---------------- TiRex ----------------
    ("m5", "tirex"): Provenance(
        "explicitly_held_out", "benchmark_exclusion", _TIREX_ZS +
        " VERIFY M5's Chronos-ZS membership."),
}

# Every remaining (dataset, model) cell is UNDOCUMENTED at source-family scope. This is a
# deliberate, explicit default and NOT a silent one: it is written into the table below so
# `provenance()` never has to invent anything at call time, and so the undocumented cells are
# countable. It is emphatically not "OOD".
for _tag in DATASETS:
    PROVENANCE.setdefault((_tag, "timesfm3"), _UNDOC_TFM3)
    PROVENANCE.setdefault((_tag, "tirex"), _UNDOC_TIREX)
    PROVENANCE.setdefault((_tag, "chronos2"),
                          Provenance("undocumented", "source_family", _T6, verified=False))
del _tag


# --------------------------------------------------------------------------- #
# lookups — every one of them fails loudly
# --------------------------------------------------------------------------- #
def _unknown(tag: str) -> KeyError:
    return KeyError(
        f"{tag!r} is not in the dataset registry. Every dataset fact (seasonal period, "
        f"display name, window builder, cluster unit) must be declared in probing/registry.py "
        f"before a driver can touch it — there is deliberately no default. Known tags: "
        f"{sorted(DATASETS)}")


def spec(tag: str) -> DatasetSpec:
    """The registry row for ``tag``. Raises ``KeyError`` on an unknown tag — never guesses."""
    try:
        return DATASETS[tag]
    except KeyError:
        raise _unknown(tag) from None


def known(tag: str) -> bool:
    """True iff ``tag`` has a registry row. For callers that must branch rather than raise."""
    return tag in DATASETS


def seasonal_m(tag: str) -> int:
    """The dataset's seasonal-naive period m, the ONLY source of it in this codebase.

    There is no default and no fallback to 24: an unknown tag raises. A global 24 applied to
    5-minute data would silently build a two-hour "season" and report a wrong MASE without
    crashing, which is exactly the failure this function exists to make impossible.
    """
    return spec(tag).seasonal_m


def display_name(tag: str) -> str:
    """The figure/table label. One spelling per dataset, everywhere.

    Before the registry this existed as SHORT / PRETTY / TITLES / EPS_TITLES / SHORT_LABELS in
    eight different modules, which had already drifted: "Uber" vs "Uber TLC", "WindFarms" vs
    "Wind Farms", "SG-Carpark" vs "SG Carpark". Those are now all this function.
    """
    return spec(tag).display_name


def slug(tag: str) -> str:
    """Filesystem-safe short name, for figure/artifact filenames."""
    return spec(tag).slug


def builder(tag: str) -> str:
    """Which window constructor this dataset uses — the replacement for ``tag in PT_OOD_TAGS``.

    The window protocol is a property of the DATASET (how its series are organized), never of
    a model's pretraining corpus.
    """
    return spec(tag).builder


def cluster_unit(tag: str) -> str:
    """The bootstrap unit: ``series``, or the parent cluster for the multi-series rosters."""
    return spec(tag).cluster_unit


def max_series(tag: str) -> int | None:
    """The deterministic series-level cap, or None for no cap."""
    return spec(tag).max_series


def role(tag: str) -> str:
    """``"primary"`` or ``"control"``."""
    return spec(tag).role


def is_control(tag: str) -> bool:
    """True for a dataset that holds something fixed instead of adding domain diversity.

    A control must be labelled as such in every table and figure and may never be counted
    toward domain coverage.
    """
    return spec(tag).role == "control"


def roster(name: str) -> list[str]:
    """A named roster: ``paper14``, ``paper7``, ``alternate``, ``legacy``, ``all``."""
    if name == "paper14":
        return list(PAPER14)
    if name == "paper7":
        return list(PAPER7)
    if name == "all":
        return list(DATASETS)
    members = [t for t, s in DATASETS.items() if s.roster == name]
    if not members:
        raise ValueError(f"unknown roster {name!r}; known: paper14, paper7, alternate, legacy, all")
    return members


def tags(roster_name: str = "paper14") -> list[str]:
    """Alias of :func:`roster` for call sites that read better as ``tags(...)``."""
    return roster(roster_name)


def primary_datasets(roster_name: str = "paper14") -> list[str]:
    """Roster members that count toward domain diversity (controls excluded)."""
    return [t for t in roster(roster_name) if not is_control(t)]


def control_datasets(roster_name: str = "paper14") -> list[str]:
    """Roster members that are controls."""
    return [t for t in roster(roster_name) if is_control(t)]


def provenance(tag: str, model: str) -> Provenance:
    """Model-relative pretraining provenance for one dataset.

    ``model`` is REQUIRED: there is no model-independent pretraining status. A dataset can be
    in one model's corpus and held out of another's, which is precisely why the old global
    ``PT_ID_TAGS`` / ``PT_OOD_TAGS`` (defined relative to Chronos-2 alone) could not be carried
    onto TimesFM-3 or TiRex.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; known: {MODELS}")
    spec(tag)                                     # fail loudly on an unknown tag first
    return PROVENANCE[(tag, model)]


def unverified_provenance(roster_name: str = "paper14",
                          models: Iterable[str] = MODELS) -> list[tuple[str, str]]:
    """The (tag, model) cells whose citation has NOT been checked against the primary source.

    These are usable as working assumptions but must not be cited in the paper until a human
    checks them. Keeping the list computable is what stops an unverified claim from quietly
    becoming a published one.
    """
    return [(t, m) for t in roster(roster_name) for m in models
            if not provenance(t, m).verified]
