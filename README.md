# Breast Cancer Feature Selection

**Binary GA/GWO feature selection and reproducible malignant-recall evaluation** on the Breast Cancer Wisconsin (Diagnostic) dataset.

> **Status: planned implementation.** This repository is the Project 2 starting point; no model performance, experiments, or test passes are claimed yet. Codex will replace these notes with verified setup instructions and results as the implementation is completed.

## Research question

Can metaheuristic feature selection reduce the number of diagnostic features while preserving classification performance, especially **recall for malignant cases**, compared with conventional feature-selection baselines?

## Planned methods

- All features; `SelectKBest(k=10)`; binary Random Search; Binary GA; Binary GWO with S-shaped and V-shaped transfer rules.
- Logistic Regression with fold-local standardization.
- Nested cross-validation on development data and one untouched final holdout test.
- Malignant recall and false negatives as primary outcomes, plus F1, ROC-AUC, precision, feature count, runtime, and evaluation counts.
- **Reuse of Project 1's actual installed package**, [`metaheuristics-lab`](https://github.com/Ehsan-Rezaeii/metaheuristics-lab), pinned to an immutable commit.

See [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) for the proposed experimental contract and `AGENTS.md` for Codex instructions. The implementation will add real run commands, CI details, saved results, and limitations here.

## Dataset and citation

Breast Cancer Wisconsin (Diagnostic): 569 rows, 30 numerical predictors, and two classes. Original source: [UCI dataset 17](https://archive.ics.uci.edu/dataset/17/breast-cancer); DOI [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). The UCI record lists a **CC BY 4.0** dataset license. The study will use the copy bundled with scikit-learn and retain explicit dataset attribution.

## Safety and scope

**Educational research only. Not for clinical diagnosis or patient care.** No claims of medical validity or model superiority are made without measured evidence.
