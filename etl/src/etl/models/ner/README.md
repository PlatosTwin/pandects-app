# NER Model

This directory is the NER root and contains the training, experiment suite, and inference entrypoints.
All paths are anchored to this root, and commands can run from any working directory.

Notes:
- JOB is SLURM_JOB_ID if set, otherwise "local".
- Set PANDECTS_REPO_ROOT to this directory if auto-detection fails.

## Pipeline (tune -> experiments -> final)

### 1) Hyperparameter tuning (Optuna, val-only)

Purpose: search hyperparameters on TRAIN and select on VAL only. The test set is never used during hyperparameter optimization—only the validation set is used for selection. Evaluation runs on the validation set during tuning.

Required:
- None (uses JOB="local" if SLURM_JOB_ID is not set).

Command:
```
python ner.py tune --num-trials 10 --max-epochs 10
```

To run in HPC:
```
sbatch run-ner.sbatch
```

Input:
- `data/ner-data.csv` (override with `--data-csv`)

Outputs:
- `configs/optuna_best_config.yaml`
- `eval_metrics/{JOB}/ner_trials/trial_###.yaml`
- `configs/splits/ner-page-splits-<split_version>.json` (created if missing)
- `model_files/ner-model-latest.ckpt` (checkpoint from retraining with best hyperparameters; overwritten by final-train)

Notes:
- **Split generation**: The split manifest is automatically created if it doesn't exist. To generate a new split, either:
  - Use a new `--split-version` name (e.g., `--split-version v2`) to create `ner-page-splits-v2.json`
  - Or delete the existing split file to regenerate with the same version name
- Splits use 10% val, 10% test, 80% train, stratified by announcement window, tagged_section, and tagged_article
- All rows with `article_upsample=True` are assigned to training only

### 2) Experiment suite (grid rows, val-only by default)

Purpose: run the experiment grid with frozen hyperparameters from `configs/optuna_best_config.yaml`.

Required:
- `configs/optuna_best_config.yaml` must exist.
- `--git-commit` is optional; recommended on HPC if .git is unavailable.

Grid file:
- `configs/grid.csv`

Single row command (explicit row id):
```
python ner.py grid-row --row-id 12 --split-version <version> --git-commit <hash>
```

Array-style command (uses SLURM_ARRAY_TASK_ID if `--row-id` is omitted):
```
python run_from_grid.py
```

To run in HPC:
```
sbatch --array=0-13 run-ner-grid-array.sbatch
```

Outputs (per run):
- `eval_metrics/{JOB}/runs/{run_id}/config.yaml`
- `eval_metrics/{JOB}/runs/{run_id}/metrics.yaml`
- `eval_metrics/{JOB}/runs/{run_id}/metrics.json`
- `eval_metrics/experiments.csv`

Notes:
- grid-row defaults to `--eval-split val`.
- **Important**: If you used a custom `--split-version` in step 1, you must pass the same `--split-version` here (defaults to `v1_article_stratified` if not specified).

### 3) Final training (train full run, test eval)

Purpose: train a final model with selected knobs and frozen hyperparameters.

Required:
- `configs/optuna_best_config.yaml` must exist.
- `--git-commit` is optional; recommended on HPC if .git is unavailable.

Command (train + test eval):
```
python ner.py final-train --train-docs 7000 --article-weight 3 --gating-mode <raw|regex|regex+snap> --split-version <version> --git-commit <hash>
```

To run in HPC:
```
sbatch run-ner-final.sbatch
```

Outputs:
- `model_files/ner-model-latest.ckpt`
- `eval_metrics/final/config.yaml`
- `eval_metrics/final/metrics.yaml`
- `eval_metrics/final/metrics.json`
- `eval_metrics/final/run_id.txt` (optional)

Notes:
- **Important**: If you used a custom `--split-version` in step 1, you must pass the same `--split-version` here (defaults to `default` if not specified). Using different split versions across steps will produce inconsistent results.
- Use the `--gating-mode` selected via validation.
- Gating/snapping is applied only at eval time; it does not change training.
- Final training always writes test metrics (`--eval-split test`).
- Use `--train-docs all` to train on the full train split (equivalent to 0).

## Demo inference (sample texts)

Purpose: run inference on `data/ner_samples.yaml` only (no training).

Command:
```
python ner.py infer-samples --ckpt-path model_files/ner-model-latest.ckpt
```
