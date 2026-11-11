# Breast Cancer Feature Selection

[![CI](https://github.com/Ehsan-Rezaeii/breast-cancer-feature-selection/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Ehsan-Rezaeii/breast-cancer-feature-selection/actions/workflows/ci.yml)

**A reproducible comparison of binary metaheuristic feature selection and simple machine-learning baselines, with particular attention to detecting malignant cases.** This study uses the [Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast-cancer), nested cross-validation, and an explicitly frozen final evaluation.

**Research question:** Can Binary Genetic Algorithm (GA) and Binary Grey Wolf Optimizer (GWO) reduce the number of predictors while preserving malignant-class recall?

**Main finding:** Binary GWO-S selected **15 of 30 features** and detected **40 of 42 malignant cases** in the held-out set—the same recall and confusion matrix as the all-features baseline. Other binary searches did not consistently preserve recall. This is **one small experimental comparison, not evidence of clinical equivalence or general superiority**.

> **Research and educational use only.** This repository is not a diagnostic tool, medical device, or recommendation for patient care.

## Results at a glance

The study uses **569 samples, 30 numeric predictors**, and a stratified **455-row development / 114-row holdout** split. The holdout contains **42 malignant** and **72 benign** cases. Malignant is defined as the positive class (`1`), so a **false negative (FN)** is a malignant case predicted as benign.

The following are **measured results from the frozen final holdout**, not cross-validation averages:

| Method | Selected features | Malignant recall | F1 | ROC-AUC | False negatives |
|---|---:|---:|---:|---:|---:|
| All features | 30 | **0.9524** | **0.9639** | **0.9987** | **2** |
| SelectKBest (k=10) | 10 | 0.9524 | 0.9412 | 0.9785 | 2 |
| Binary GA | 14 | 0.9048 | 0.9383 | 0.9947 | 4 |
| **Binary GWO-S** | **15** | **0.9524** | **0.9639** | 0.9891 | **2** |
| Binary GWO-V | 9 | 0.9048 | 0.9383 | 0.9848 | 4 |
| Binary Random Search | 16 | 0.9286 | 0.9398 | 0.9940 | 3 |

**Interpretation:** GWO-S used half as many features as the all-features baseline and produced the **same aggregate holdout confusion-matrix counts** (TN=71, FP=1, FN=2, TP=40). Identical confusion-matrix counts do not prove that the models predicted the same individual samples; GWO-S also had a lower ROC-AUC. These findings do **not** establish predictive equivalence or clinical safety. SelectKBest achieved the same recall with only 10 features but made more false-positive predictions. GA and GWO-V each missed two additional malignant cases.

![Development CV: malignant recall versus selected feature count](results/full/development_recall_features.png)

See [complete measured results](docs/RESULTS.md) for outer-fold means and variability, final confusion matrices, selected masks, search costs, and limitations. The [holdout confusion-matrix figure](results/full/final_confusion_matrices.png) is also available.

## Experimental design

The predefined protocol separates model development from final evaluation:

1. **Data and target:** scikit-learn's bundled Wisconsin Diagnostic dataset; original labels are explicitly remapped so `1 = malignant` and `0 = benign`.
2. **Evaluation split:** stratified 80/20 development/holdout partition (`random_state=2026`), followed by **3-fold outer CV** on development data (`2027`) and **3-fold inner CV** for each training fold (`2028`).
3. **Baselines:** all 30 features and `SelectKBest(f_classif, k=10)`.
4. **Search methods:** Binary GA, Binary GWO with S-shaped and V-shaped transfer rules, and binary Random Search. Each uses **population 12** and exactly **90 objective evaluations** per search. All four optimizers come from the [commit-pinned `metaheuristics-lab` library](https://github.com/Ehsan-Rezaeii/metaheuristics-lab/tree/6b4ef0fa3d72f608146f3dd7eabb84bf39fa896e); no optimizer code is copied.
5. **Classifier:** fold-local `StandardScaler` and `LogisticRegression(C=1, solver="liblinear", max_iter=2000, random_state=0)`, with a fixed 0.5 decision threshold.
6. **Fitness:** minimize `0.70 × (1 − mean recall) + 0.30 × (1 − mean F1) + 0.02 × selected/30`. An empty mask receives a fixed penalty and fits no model.
7. **Finalization:** repeat full-development searches with seeds `0, 1, 2`; select masks using development-only fitness and predefined tie-breaks; freeze masks and settings before the full-study holdout evaluation.

Preprocessing, feature selection, and fitted classifiers remain inside their permitted training folds. **Optimized inner-CV scores, outer-CV development estimates, and final holdout results are reported separately.**

All four search methods were allocated equal objective-call budgets, **not** equal running time. Across the full study, the saved audit reports **2,160 search objective calls and 6,508 classifier fits** (including baseline and final fits). The low overlap between repeated selected masks is documented rather than concealed.

For exact fold assignments, seeds, optimizer configurations, freeze checksums, and timing definitions, see the [experimental protocol](docs/EXPERIMENT_PROTOCOL.md) and [results](docs/RESULTS.md).

## Install and verify

**Requirements:** Python 3.11+ and Git (CI verifies Python 3.11 and 3.12). The installed Project 1 dependency is fetched from a specific Git commit; network access is needed for initial installation.

```bash
git clone https://github.com/Ehsan-Rezaeii/breast-cancer-feature-selection.git
cd breast-cancer-feature-selection
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run the fast checks without repeating the full study:

```bash
ruff check .
ruff format --check .
pytest -q
python experiments/smoke_test.py
python experiments/audit_results.py results/full
python -m build
```

The smoke test uses synthetic data; it is **not** a second measurement on the Wisconsin holdout. The committed GitHub Actions workflow runs checks, artifact auditing, packaging, and a clean wheel import on Python **3.11 and 3.12**. The recorded full-study run and validation evidence are described in [RESULTS.md](docs/RESULTS.md).

## Reproduce the study

Use a **new output directory** and run from the repository root. For reproducible single-threaded numerical execution, set the following environment variables on Linux/macOS:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/p2-mpl XDG_CACHE_HOME=/tmp/p2-cache

bcfs-study develop --output results/reproduction
# Review development artifacts and the frozen choices before final evaluation.
bcfs-study final --output results/reproduction
bcfs-study report --output results/reproduction
python experiments/audit_results.py results/reproduction
```

The staged workflow refuses a second final evaluation in the same directory. Re-running the study verifies computational reproducibility; **it is not a new independent clinical validation**, because the final split has already been observed. Durations and rendered image bytes may differ across systems. The recorded scientific dependency versions are in `pyproject.toml`, with the measured environment snapshot in [`results/full/requirements_environment.txt`](results/full/requirements_environment.txt).

## Evidence and repository guide

| Resource | Contents |
|---|---|
| [Study specification](docs/PROJECT_SPEC.md) | Research question, scope, and acceptance criteria |
| [Experimental protocol](docs/EXPERIMENT_PROTOCOL.md) | Frozen splits, budgets, objective, and safeguards |
| [Measured results](docs/RESULTS.md) | CV/holdout tables, compute costs, variance, and candid interpretation |
| [Implementation walkthrough](docs/IMPLEMENTATION_WALKTHROUGH.md) | Code structure and methodological explanations |
| [`results/full/`](results/full/) | Fold indices, search traces, masks, freeze files, results, figures, and checksums |
| [`src/breast_cancer_feature_selection/`](src/breast_cancer_feature_selection/) | Installable data, evaluation, selection, and experiment modules |
| [`tests/test_study.py`](tests/test_study.py) | Data invariants, budget checks, leakage regressions, and freeze guards |

## Limitations and research integrity

- **Small, historical dataset:** 42 malignant cases in one final test split are too few to establish robust clinical performance, equivalence, or statistical superiority. One additional false negative changes malignant recall by about **2.38 percentage points**.
- **No external or prospective validation:** feature selection results must not be interpreted as patient-level diagnostic evidence.
- **Feature-mask instability:** the three development seeds selected different correlated subsets; mask frequency does not imply biological causality.
- **Fixed scope:** one Logistic Regression configuration, one threshold, short search budgets, and three outer folds limit the conclusions. Search methods are **not assumed to beat simpler baselines**.
- **Procedural disclosure:** before the registered full study, a preliminary automated one-call pilot exercised the real-data final code path. Its scores were **not inspected or used for decisions**. Thus the subsequent mask and model choices were development-only, but it would be inaccurate to claim that the holdout was never touched by any preflight code. See [the full explanation](docs/RESULTS.md#limitations-and-procedural-caveat).

No tuning was performed in response to the reported final results. The full-study masks and configurations were frozen before the recorded final evaluation; its results cannot now be treated as a fresh confirmatory test.

## Dataset attribution and project context

**Dataset:** Wolberg, Mangasarian, Street, and Street (1993), *Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository. [UCI record](https://archive.ics.uci.edu/dataset/17/breast-cancer) · [DOI: 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B) · [dataset license: CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset is accessed through `sklearn.datasets.load_breast_cancer`; no separate raw dataset is committed.

This is the **feature-selection and malignant-recall study** in a five-project research portfolio. It installs [metaheuristics-lab](https://github.com/Ehsan-Rezaeii/metaheuristics-lab) as a commit-pinned dependency instead of copying optimizer code. Related applications investigate [software defect prediction on Apache Camel 1.4 to 1.6](https://github.com/Ehsan-Rezaeii/software-defect-prediction), [SQL injection detection](https://github.com/Ehsan-Rezaeii/sqli-detection-lab), and [search-based test generation](https://github.com/Ehsan-Rezaeii/search-based-test-generation). These studies use different datasets, objectives, and metrics; their numerical results are not directly comparable.

**Code license:** [MIT](LICENSE). Dataset rights and attribution are separate from the code license.

