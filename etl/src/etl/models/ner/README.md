# NER Model

This directory contains the local NER training and evaluation pipeline for structural agreement tagging:

- `ARTICLE`
- `SECTION`
- `PAGE`

The current supported pipeline is:

- parquet input by default
- fixed split manifests by `split_version`
- `boundary_mix` sampling only
- raw metrics only
- validation-set experiment sweeps
- test-set evaluation only for the final selected recipe

## Commands

### Hyperparameter tuning

Runs Optuna on train/val only.

```bash
python ner.py tune --num-trials 10 --max-epochs 10
```

HPC:

```bash
sbatch run-ner.sbatch
```

Primary outputs:

- `configs/optuna_best_config.yaml`
- `eval_metrics/{JOB}/ner_trials/...`
- `configs/splits/ner-page-splits-<split_version>.json`
- `model_files/ner-model-latest.ckpt`

### Experiment sweep

Runs one row from the current experiment grid against the validation set.

```bash
python ner.py grid-row --row-id 0 --split-version <version> --git-commit <hash>
```

Array entrypoint:

```bash
python run_from_grid.py
```

HPC:

```bash
sbatch run-ner-grid-array.sbatch
```

Primary outputs:

- `eval_metrics/{JOB}/runs/{run_id}/config.yaml`
- `eval_metrics/{JOB}/runs/{run_id}/metrics.yaml`
- `eval_metrics/{JOB}/runs/{run_id}/metrics.json`
- `eval_metrics/experiments_xp.csv`

Notes:

- `configs/grid.csv` is the active experiment matrix.
- `configs/frozen_experiment_config.yaml` is the fixed training recipe used by the grid.
- Keep the same `split_version` across tuning, experiment sweeps, and final training.

### Final training

Trains the selected recipe and evaluates it on the test split.

```bash
python ner.py final-train --train-docs all --decoder-mode independent --split-version <version> --git-commit <hash>
```

Add the winning architectural flags from validation to that command. For example, `--boundary-head`, `--boundary-loss-weight`, `--preserve-case`, or a different `--decoder-mode`.

HPC:

```bash
sbatch run-ner-final.sbatch
```

`run-ner-final.sbatch` is a template. Set `FINAL_FLAGS`, `SPLIT_VERSION`, and `GIT_COMMIT` before submission so the final run matches the chosen validation recipe exactly.

Primary outputs:

- `model_files/ner-model-latest.ckpt`
- `eval_metrics/final/config.yaml`
- `eval_metrics/final/metrics.yaml`
- `eval_metrics/final/metrics.json`

### Audit article failures

Exports problematic `ARTICLE` cases from a trained checkpoint. This does not retrain the model.

```bash
python ner.py audit-articles --ckpt-path /path/to/best.ckpt --eval-split val --split-version <version> --output-path logs/article_audit_cases.jsonl
```

Use this after a validation run to inspect strict-boundary failures quickly.

## Data And Splits

- Default data path: `data/ner-data.parquet`
- If a split manifest for the requested `split_version` does not exist, it is created automatically.
- If it does exist, it is reused as-is.
- For experiment comparisons, reuse the same `split_version`.

## Artifacts

This directory intentionally keeps active runtime artifacts nearby:

- `logs/`
- `eval_metrics/`
- `model_files/`

## Current Interface

Supported public commands:

- `tune`
- `grid-row`
- `final-train`
- `recover-run`
- `audit-articles`

## References

- Current status: [CURRENT_STATE.md](/Users/nikitabogdanov/PycharmProjects/merger_agreements/appv2/etl/src/etl/models/ner/CURRENT_STATE.md)
