# AGENTS.md — Codex instructions for `breast-cancer-feature-selection`

## Mission and authority

This repository is **Project 2 of a five-repository, research-oriented portfolio**. Its research question is: **Can binary metaheuristic feature selection reduce the number of features while preserving classification performance, particularly malignant-class recall, on the Breast Cancer Wisconsin (Diagnostic) dataset?** The study is methodological and educational; **it is not a medical device or a clinically validated diagnostic tool**.

Read `docs/PROJECT_SPEC.md` **before implementation**. It is the concrete Project 2 experimental contract. The portfolio-wide `PROJECT_BLUEPRINT.md`, when available, defines approved project scope; never silently replace it. Explicit user instructions can revise the workflow. Do not implement Projects 3–5 here.

The user wants **implementation first, teaching later**. Work through the complete approved Project 2 scope, with tests and evidence at internal milestones. Prepare a detailed English technical walkthrough for a later, separate Persian learning session; do not stop after each file to teach. Do not say anything is tested or measured until it actually is.

## Mandatory reuse of Project 1

`metaheuristics-lab` must be an **actual installable dependency**, not copied code, a submodule containing vendored implementations, or a rewritten optimizer. Use the public API:

```python
from metaheuristics_lab import BinaryGA, BinaryGWO, Problem, RandomSearch

problem = Problem.binary(objective, dimension=30)
result = BinaryGA(population_size=12).optimize(
    problem, max_evaluations=90, seed=0
)
mask = result.best_solution.astype(bool)
```

This API was checked in Project 1 at commit `6b4ef0fa3d72f608146f3dd7eabb84bf39fa896e` in `Ehsan-Rezaeii/metaheuristics-lab`. Pin this **immutable commit** in the dependency specification or a reproducible installation mechanism; never assume that the package is on PyPI. Verify installation, actual imports, and the recorded dependency revision. Do not silently modify Project 1 to simplify Project 2.

## Study design: preserve scientific validity

1. Load the original Breast Cancer Wisconsin **Diagnostic** data using `sklearn.datasets.load_breast_cancer`; document UCI origin, citation, DOI, and dataset license. Verify 569 rows, 30 numerical predictor features, and class counts.
2. **Explicitly remap malignant to positive label `1`.** The scikit-learn source uses malignant=`0`, benign=`1`; write tests to enforce this mapping. All reported positive-class recall, F1, probabilities, and ROC-AUC must correspond to **malignant**.
3. Make one predeclared **stratified 80/20 development/test split** with a fixed seed (`2026` proposed in the spec). Keep all test rows, labels, and derived statistics completely outside feature-selection fitness, hyperparameter decisions, model/seed selection, and visualization during development.
4. Model: fixed **Logistic Regression** with `StandardScaler` in a scikit-learn `Pipeline` (see exact defaults in `PROJECT_SPEC.md`). Do not add unapproved models or threshold-tuning experiments as though they were required.
5. Compare **AllFeatures**, **SelectKBest**, **BinaryGA**, **BinaryGWO-S**, **BinaryGWO-V**, and **binary RandomSearch**. Use the *same training data, inner CV splits, classifier, fitness, and evaluation budget* for binary search methods. Time and count classifier fits separately; feature-filter baselines do not have to consume artificially equal search budgets.
6. Fit all supervised preprocessing/selection steps inside the allowed training folds. For metaheuristics, rerun feature selection entirely within **each outer training fold** before measuring outer-fold performance; it is invalid to select a mask on the entire development set and then claim its CV score is unbiased. Inner CV is for mask search; outer CV is for development-stage performance estimation. Neither may use the final test set.
7. Fix the fitness function, empty-mask penalty, feature-count trade-off, evaluation budget, threshold, seeds, and baselines **before** opening final-test results. The spec gives conservative defaults; document any justified change before measuring.
8. At the end, freeze a *single final configuration and selected mask per method* using only development data. Fit that method's classifier on the development data and **evaluate each frozen method once** on the held-out test set. Do not pick winners or revise the experiment after inspecting test outcomes. Record test confusion matrix and number of malignant false negatives.
9. Report malignant recall, F1, ROC-AUC, precision, confusion matrix, feature count, runtime, and search/fit counts. Distinguish outer-CV estimates from one-time final holdout results. On this small dataset, do **not** present perfect recall or a difference of a few errors as clinical validity.
10. Multiple optimizer seeds measure search variability, **not** independent patient cohorts or independent test-set replications. Avoid artificial significance from reusing a single held-out split. Negative or mixed results are welcome.

## Correctness, tests, and reproducibility

- Test labels, counts, mask bit convention (`1 = selected`), empty-mask handling, metric positive class, the ROC-AUC probability column, fit-on-train-only behavior, and no access to holdout during tuning. Use at least one deliberate leakage-regression test (e.g., perturb held-out labels and demonstrate that frozen selection is unchanged).
- Assert exact optimization objective-call budgets and record inner-CV model fits separately. Use independent NumPy seeds, consistent population sizes and objective budgets across GA/GWO/RandomSearch, and document any limitation of equal evaluation budgets as a proxy for compute.
- Keep train/test indices, seed lists, versions, the Project 1 dependency commit, configs, raw per-run measurements, summary tables, confusion matrices, and plots reproducible. Include actual timing methodology and explicitly mark pilot vs final experiments.
- Keep CI fast (`pytest`, lint/format, install/import smoke checks). Longer nested-CV experiments should be opt-in commands, not run on every PR. Use `error_score='raise'` for experiments; do not hide model failures.
- Prefer stable APIs, simple modular Python, type hints, `pathlib`, and small tested functions. Use an installable `src/` package. Never commit secrets, virtualenvs, raw patient-level records unnecessary for reproducibility, large caches, or fabricated results.
- Verify license compliance: code license is separate from dataset provenance and attribution. The built-in scikit-learn dataset does not require storing a duplicate raw download in this repository.

## Suggested layout

```text
breast-cancer-feature-selection/
  AGENTS.md
  README.md
  pyproject.toml
  LICENSE
  .gitignore
  docs/PROJECT_SPEC.md
  docs/EXPERIMENT_PROTOCOL.md
  docs/RESULTS.md
  docs/IMPLEMENTATION_WALKTHROUGH.md
  src/breast_cancer_feature_selection/
  tests/
  experiments/
  results/
  .github/workflows/
```

Adapt this layout to existing files without unnecessary reorganization. Keep optimization implementations inside Project 1's distribution.

## Execution and final handoff

1. Inspect current repository and the spec. Draft a concise implementation plan. Build an installable skeleton and baseline first.
2. Implement leakage-safe nested development evaluation, binary adapters to Project 1, baseline comparison, and one-time frozen final evaluation.
3. Run tests, lint, formatting checks, package build/install, small smoke experiments, then the full feasible protocol. Recheck all generated tables and figures against raw rows.
4. Document accurate installation, full/quick commands, data attribution, method, hypotheses, measured results, uncertainty, and limitations. Put important research keywords first in the GitHub README description.
5. Prepare the work on a feature branch and open a PR. **Do not merge or force-push automatically**; preserve the main branch for review. If permissions block PR creation, report the exact blocker.
6. Deliver a final report separating: implemented, actually tested, actually measured, and remaining limitations. Include commands and observed outcomes; do not invent success.

**Done means:** the full approved Project 2 study is reproducible, its test set was kept untouched until the final frozen evaluation, there is direct dependency reuse of Project 1, all required baselines and malignant-centric metrics are reported, and the code/tests/docs agree. A green badge alone is not proof of methodological correctness.
