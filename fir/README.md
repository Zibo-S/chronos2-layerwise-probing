# Running the rebuttal phases 2c / 2d / 2e / 2f on Fir (H100)

Fir runs the same code, protocol and inputs as the Narval jobs. Only the cluster-specific job
layer changes. The Narval jobs (9229954 2c, 9229962 2d, 9229964 2e, 9253740 2f) are a separate
copy and are never touched from here.

## Layout (identical paths on both clusters, separate filesystems)

| what | Fir path |
|---|---|
| repo clone (branch `rebuttal`) | `/scratch/eserebri/repos/chronos2-layerwise-probing` |
| venv (rebuilt with `fir/setup_fir_venv.sh`, never copied) | `/scratch/eserebri/venvs/c2lp-py311` (`.venv`) |
| Chronos-2 K=4 forecast-slot caches (75 files) | `/scratch/eserebri/chronos2/features_cache` (`features_cache`) |
| Phase-1 windows, TimesFM-3 / TiRex caches, OOD shards | `/scratch/eserebri/chronos2/{phase1_shared_windows,phase1_shared_cache,ood_targets}` |
| H3 Chronos-2 adapters | `/scratch/eserebri/chronos2_phase2/adapters/chronos2` |
| HF checkpoints (offline in jobs) | `/scratch/eserebri/chronos2/hf_cache/hub/models--*` |

The inputs were copied from Narval with `fir/transfer_inputs_from_narval.sh`, which streams
Narval → Mac → Fir (both clusters need MFA) and checks the stream's sha256 on both ends and
every file's size. The drivers validate the inputs again in every job: window digests, cache
labels, and adapter sha256 against the committed H3 records.

## Jobs

`bash fir/submit_fir.sh smoke`, then `bash fir/submit_fir.sh full [--after <smoke id>]`, from the
repo root on a Fir login node. Every job is ≤ 3 h, the shortest Fir GPU tier, on one full H100
(`--gpus-per-node=h100:1`). The chunks partition each phase's cells, so no cell is computed twice:

| phase | chunks | per chunk |
|---|---|---|
| 2c LODO (48 cells) | 7 | sibling_excluded × one depth × half the targets (7 cells); sibling_included (6 cells) |
| 2f source scaling (40 cells) | 4 | one target (N = 1 cells first, then 4, 8, 13) |
| 2d pooled readouts (42 cells) | 3 | one backbone, gated by its own one-batch check + native-refit preflight |
| 2e non-contiguous (14 datasets) | 3 | 4 / 5 / 5 datasets; the latency stage in the first job only (H100 timings) |

Per phase, `fir/job_fir_aggregate.sh` (CPU) runs once after all chunks:
1. It follows timeout continuations.
2. It builds the tables and report (stamped INCOMPLETE if cells are missing).
3. It stages the checksummed archive only if every chunk ended COMPLETED with exit 0.

Resume or retry: resubmit the identical line (COMPLETE cells are skipped). The ledger
`results/fir_submissions.tsv` records the cluster, commit, job ID, name and arguments of every
submission.

## Provenance and merging

* Every job writes `results/<tree>/slurm/<jobid>/` with:
  * `job.txt`: host `fc*`, git commit;
  * `env.txt`: `RB_CLUSTER=fir`;
  * `nvidia_smi.csv`: H100;
  * `pip_freeze.txt`.
* Every driver writes `runs/<jobid>/run_manifest.json`.
* Fir job IDs (~4.0M) cannot collide with Narval's (~9.2M).
* Do NOT copy a Fir results tree over a Narval one, or the reverse. If both clusters finish a
  phase, keep one complete set per phase and record which cluster produced it.
* Commit Fir results from the Fir clone on a separate branch (`rebuttal-fir`), never on `rebuttal`.
* Hardware: the fits are FP32 with TF32 off on both clusters, and every reproduction gate
  compares Fir against the committed A100 artifacts at the unchanged tolerances. The 2e latency
  numbers are H100 timings and must be labelled as such. They are not comparable with the
  paper's A100 latency table.
