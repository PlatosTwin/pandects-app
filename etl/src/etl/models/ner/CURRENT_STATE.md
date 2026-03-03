# Current State

This NER package is currently centered on structural tagging for merger agreements:

- `ARTICLE`
- `SECTION`
- `PAGE`

Current supported modeling path:

- `boundary_mix` sampling
- raw metrics only
- parquet data input by default
- validation-set experiment selection
- final test-set evaluation through `final-train`

Current operational conventions:

- experiment sweeps append to `eval_metrics/experiments_xp.csv`
- split manifests are keyed by `split_version`
- final training should reuse the same `split_version` chosen during validation

Current cleanup status:

- deprecated artifacts moved under `etl/src/etl/models/deprecated/`
- regex/snap code removed
- legacy sampling removed
- dead postprocessing harness removed

This file is meant to describe the supported present-day state of the package, not its historical experiments.
