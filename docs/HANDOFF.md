# Project 2 handoff

Implemented and measured work is in PR #1:
https://github.com/Ehsan-Rezaeii/breast-cancer-feature-selection/pull/1
Branch: feat/project2-complete-study. No merge or auto-merge was performed.

## Verified engineering evidence

- Python 3.12.14 locally; 13 pytest tests passed (latest local run: 3.61 seconds).
- `ruff check .`: all checks passed. `ruff format --check .`: 9 files formatted.
- `python experiments/smoke_test.py`: synthetic end-to-end pilot completed.
- `bcfs-study smoke --output results/smoke-cli`: retained synthetic pilot completed.
- `python -m build`: sdist and wheel built successfully.
- Clean `/tmp/p2-wheel-check` virtual environment installed the wheel; imports of
  Problem, BinaryGA, BinaryGWO and RandomSearch plus this package succeeded while
  the current directory was `/tmp`, outside the checkout.
- Installed Project 1 metadata reports commit
  `6b4ef0fa3d72f608146f3dd7eabb84bf39fa896e`. This is an actual direct dependency.
- GitHub Actions run 42851157929 passed on Python 3.11 and 3.12 at source commit
  c4f8e0a. Run 42851736749 passed both matrix jobs at results commit a3fbcc8,
  including lint/format, tests, synthetic smoke, artifact audit, package build,
  clean wheel install and outside-checkout imports. No CI result is inferred
  solely from a badge or local execution.

Hosted result: https://github.com/Ehsan-Rezaeii/breast-cancer-feature-selection/actions/runs/42851736749

The local `gh auth status` command reported an invalid GH_TOKEN. Git transport and
the GitHub connector nevertheless successfully pushed the branch and opened the
PR. The connector's commit workflow lookup filters to PR-triggered runs and returned
none; the push-triggered run IDs were obtained from the public Actions page and
both successful jobs/steps verified with the workflow jobs API. No permissions
blocker remains for the delivered branch/PR.

## Actually executed full experiment

All commands ran from the repository root with these environment settings:

```bash
export MPLCONFIGDIR=/tmp/p2-mpl XDG_CACHE_HOME=/tmp/p2-cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/bcfs-study develop --output results/full
.venv/bin/bcfs-study final --output results/full
.venv/bin/bcfs-study report --output results/full
.venv/bin/python experiments/audit_results.py results/full
```

Develop and final each ran once for the full study. Report and audit were repeated
to validate and checksum derived outputs; they do not fit estimators. stdout is
saved under results/full. The audit reports PASS for hashes, source identity,
partitions, nesting, budgets, fits, metrics, selected masks/seeds and chronology.
The run contains 18 outer rows, 12 training-only repeat searches, 6 final rows,
2,160 objective calls, 6,480 inner fits, and 6,508 total classifier fits. All six
comparators and all specified malignant metrics were actually measured.

GWO-S used 15 features and matched AllFeatures' recall of 40/42 with two false
negatives. GA (14 features) and GWO-V (9) missed four; RandomSearch (16) missed
three; SelectKBest (10) missed two and produced three false positives. Full tables,
uncertainty, timings and stability analysis are in RESULTS.md. No clinical
performance or statistical superiority is claimed.

## Limitations and remaining issues

The approved full budget completed, with no pending implementation milestone or
external runtime blocker. An initial automated one-call real-data preflight test
exercised the holdout code path without displaying/inspecting its metrics or using
them for decisions. Subsequent end-to-end preflight checks use synthetic data.
This means literal never-processed-by-any-preflight-code holdout isolation cannot
be claimed. The full study's choices remained development-only and were committed
before full-budget final evaluation. See the protocol and RESULTS.md caveat.

Scientific limitations remain: one small historical dataset/split, three outer
folds and search seeds, low mask stability, fixed model/trade-off/budget, no
external validation, and machine-dependent timing. The test split must not be
reused for future tuning. CI is fast engineering validation, not a new scientific
replication. Patient-level predictions were not saved, so ROC-AUC is tested in code
but cannot be recomputed by the artifact-only audit.

## File inventory

A means created; M means updated. Existing AGENTS.md, PROJECT_SPEC.md and LICENSE
were retained unchanged. Existing Python ignore rules were preserved and extended.
In addition to the inventory below, this handoff document itself was created.

```text
A	.github/workflows/ci.yml
M	.gitignore
M	README.md
A	docs/EXPERIMENT_PROTOCOL.md
A	docs/IMPLEMENTATION_WALKTHROUGH.md
A	docs/RESULTS.md
A	experiments/audit_results.py
A	experiments/run_study.py
A	experiments/smoke_test.py
A	pyproject.toml
A	results/full/audit_stdout.txt
A	results/full/checksums.json
A	results/full/class_summary.json
A	results/full/config.json
A	results/full/cost_summary.csv
A	results/full/develop_stdout.txt
A	results/full/development_recall_features.png
A	results/full/development_rows.csv
A	results/full/development_summary.csv
A	results/full/environment.json
A	results/full/feature_summary.csv
A	results/full/final_confusion_matrices.png
A	results/full/final_rows.csv
A	results/full/final_stdout.txt
A	results/full/folds.json
A	results/full/freeze.json
A	results/full/freeze.sha256
A	results/full/holdout_completed.json
A	results/full/holdout_started.json
A	results/full/requirements_environment.txt
A	results/full/search_traces.json
A	results/full/seed_repeats.json
A	results/full/split.json
A	src/breast_cancer_feature_selection/__init__.py
A	src/breast_cancer_feature_selection/data.py
A	src/breast_cancer_feature_selection/evaluation.py
A	src/breast_cancer_feature_selection/experiment.py
A	src/breast_cancer_feature_selection/selection.py
A	tests/test_study.py
```

Install/reproduction commands are in README.md. EXPERIMENT_PROTOCOL.md records the
predefined design; IMPLEMENTATION_WALKTHROUGH.md explains each module for the later
learning session. results/full/checksums.json inventories the measured artifacts.
