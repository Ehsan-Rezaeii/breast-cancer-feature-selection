# Project 2 — Experimental specification (pre-implementation)

**Repository:** `breast-cancer-feature-selection`  
**Status:** *Planned protocol; no Project 2 experiments or results are claimed in this document.*  
**Research question:** Can metaheuristic feature selection reduce feature count while preserving classification performance, especially **malignant-class recall**, compared with simple baselines?

## 1. Data and task

- **Dataset:** Breast Cancer Wisconsin (Diagnostic), UCI dataset 17, accessed using `sklearn.datasets.load_breast_cancer()`. Data contain **569 samples**, **30 positive-valued numerical features**, **212 malignant** and **357 benign** labels. This is the diagnostic dataset, not the older Wisconsin (Original) dataset.
- **Target mapping:** scikit-learn's `target_names` are `['malignant', 'benign']` and its source label codes are malignant=`0`, benign=`1`. Convert with `y = (data.target == 0).astype(int)`, making **malignant=`1` / benign=`0`**. Assert this transformation in tests. If a classifier outputs two probability columns, choose the column whose `classes_` entry is `1`.
- **Features:** use only the 30 provided predictors. Never use sample IDs, row index, label, or target-derived statistics as predictors.
- **Provenance:** UCI original repository: https://archive.ics.uci.edu/dataset/17/breast-cancer ; DOI: https://doi.org/10.24432/C5DW2B ; dataset attribution: Wolberg, Mangasarian, Street, & Street (1993), *Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository. The UCI listing specifies CC BY 4.0; cite and attribute it in the README. The built-in scikit-learn copy avoids external downloads.
- **Warning:** educational methodological study only, not a clinical application or safety-critical diagnosis.

## 2. Methods and comparator fairness

Use **one primary classifier** to isolate the feature-selection effect:

- `Pipeline([('scale', StandardScaler()), ('model', LogisticRegression(C=1.0, solver='liblinear', max_iter=2000, random_state=0))])` for models using an already-chosen mask. Fit only within authorized training data. For `SelectKBest`, put selection inside the pipeline with `SelectKBest(f_classif, k=10)` preceding scaling and the classifier; `k=10` is a predeclared, non-tuned illustrative baseline, not a post-hoc winner.
- Classifier default prediction threshold: **0.5**, fixed before final evaluation. Do not tune a threshold against held-out labels.
- Methods (report exactly as separate rows): `AllFeatures`, `SelectKBest(k=10)`, `BinaryGA`, `BinaryGWO-S`, `BinaryGWO-V`, and `RandomSearch(binary)`.
- **Real dependency:** import optimizers directly from `metaheuristics_lab` (`Problem.binary`, `.optimize(problem, max_evaluations=..., seed=...)`); do not copy or rewrite optimization algorithms. Pin source dependency to git commit `6b4ef0fa3d72f608146f3dd7eabb84bf39fa896e` from `https://github.com/Ehsan-Rezaeii/metaheuristics-lab`. Its package distribution is `metaheuristics-lab` and import name `metaheuristics_lab`; no PyPI release is assumed.

### Search fitness (minimize; predefined, not inferred from final test)

For each nonempty 30-bit mask `m`, evaluate only the **inner-training** subset through stratified inner cross-validation:

```
fitness(m) = 0.70 * (1 - mean_inner_malignant_recall)
           + 0.30 * (1 - mean_inner_malignant_F1)
           + 0.02 * (selected_features / 30)
```

An all-zero mask has fitness `2.0` (explicit invalid-mask penalty; do not fit an estimator). Larger feature penalties or different weights must be registered in the method document **before** final holdout evaluation. The primary formula prioritizes malignant recall while penalizing false positives indirectly through F1; it is a declared experimental design choice, not an optimal clinical cost function. Report an all-negative-predictions sanity comparison if useful, but do not present it as an approved main comparator.

Every mask objective call counts as one optimizer **objective evaluation** even when the mask repeats. Maintain a separate counter of actual classifier fits; if memoization is introduced, distinguish optimizer calls from saved model fits transparently. Record masks and counts.

### Resource-conscious fixed defaults

- Development / final test split: **80/20 stratified**, `random_state=2026`, indices saved. No preprocessing before splitting.
- Development estimate: **3-fold stratified outer CV**, shuffled with `random_state=2027`. Within each outer training fold, run a **fresh 3-fold stratified inner CV**, shuffled with `random_state=2028`. The optimizer is rerun separately in each outer fold. Use predeclared seed per outer fold (`0,1,2`) and include actual seed mapping in results. These three outer folds measure split variability but are **not independent clinical cohorts**.
- Binary optimization budget: **90 objective evaluations per method per outer fold**, population size `12`, one seed per outer fold. Search methods use identical inner CV indices within a fold and identical classifier settings. Use a very small smoke configuration for CI; full experiments run manually.
- For **within-development search-seed stability**, optionally run `seed in [0, 1, 2]` on the complete development set with its own 3-fold training-only CV and the same budget; report these as *development-stage exploratory repeats*, **not** three test-set replications. Select the one final mask per search method by the lowest training-CV fitness, with a documented deterministic tie-break, before accessing the held-out test rows.
- Train AllFeatures and SelectKBest on the full development set. Fit the final selected-mask logistic classifier for each binary search method on the full development set.
- Evaluate the **one frozen result per method** exactly once on the held-out test set; record a timestamp or manifest documenting the freeze (mask, config hash, seed, objective, intended test metrics) before evaluating. Never tune or choose seeds by test results.
- Adjust the resource budget only before final-test evaluation, and record the amendment with a reason. A pilot configuration must be explicitly labeled, never passed off as a final protocol.

### Interpreting CV correctly

The *inner* CV fitness is deliberately optimized and is therefore **not an unbiased estimate of generalization**. The *outer* CV estimates a complete selection-plus-fitting process, since feature selection is re-run inside each outer training fold. The independent held-out final test gives one additional estimate for the frozen, final choices. The test must not guide subsequent iterations of this study. See scikit-learn's nested CV documentation.

## 3. Outcomes, reporting, and interpretation

**Primary outcome:** malignant-class **Recall** = TP/(TP+FN), with malignant as `pos_label=1`. Report **false negative count (FN)** explicitly, along with the malignant support (count of true malignant test examples).

**Secondary outcomes:** malignant F1, malignant precision, ROC-AUC computed from score/probability for class 1, confusion matrix (row order `[1,0]` is acceptable if clearly documented), number/percentage of selected features, runtime, number of objective evaluations, number of model fits. Accuracy can be shown, but must not displace recall or FN analysis. Include a plot of development CV performance versus feature count and optional distribution/stability plots; mark the stage for every plotted metric.

Do not claim clinical usefulness, statistical superiority, sensitivity improvements, or equivalent safety based on a few cases or a single test split. Negative findings and uncertainty must be reported.

## 4. Engineering and verification

Expected repository deliverables:

```text
AGENTS.md
README.md
pyproject.toml
LICENSE
.gitignore
src/breast_cancer_feature_selection/
  __init__.py
  data.py
  evaluation.py
  selection.py
  experiment.py
experiments/
  run_study.py
  smoke_test.py
tests/
docs/
  PROJECT_SPEC.md
  EXPERIMENT_PROTOCOL.md
  RESULTS.md
  IMPLEMENTATION_WALKTHROUGH.md
results/
.github/workflows/ci.yml
```

The tree is guidance, not an inflexible requirement. Implement a small CLI and a `--smoke` mode with a much smaller budget; full nested CV should not run in CI. Keep code modular and installable; use `pathlib`, type hints, and clear public APIs. Save configs, git/dependency revision and environment versions, development CV rows, selected masks, confusion matrices, final holdout rows, and figures. Exclude caches and bulky data.

**Test acceptance** should include: dataset/source label mapping; no duplicate/leaked train/test indices; masked classifier is fit from training subsets; nested selector fits only inside outer training data; held-out label perturbation cannot alter frozen masks/configurations; empty masks never fit; equal optimizer objective-call budgets; exact malignant recall/F1/ROC-AUC semantics; deterministic seeds; proper Project 1 dependency import; fast end-to-end smoke command; CI lint, tests, packaging and import checks.

## 5. Milestones and completion

1. Bootstrap repository, package/dependency, dataset inspection and source attribution.
2. Implement baseline pipelines, correct malignant metrics, split/fold enforcement, tests.
3. Integrate BinaryGA and BinaryGWO-S/V plus RandomSearch directly from Project 1, add fairness/accounting and leak tests.
4. Produce outer-CV study and training-only stability analysis; inspect methodology before exposing final test.
5. Freeze final configurations, evaluate held-out test once, save genuine results and audit tables/plots.
6. Add CI, documentation, limitations, implementation walkthrough; create PR for independent review. Do not merge automatically.

**Definition of done:** authentic experiment artifacts and successful tests support all factual claims; no leakage or test reuse for tuning; comparators and outcome metrics are complete; the package genuinely depends on Project 1; and the PR is ready for independent review.

## Reference links (verify before quoting further)

- UCI: https://archive.ics.uci.edu/dataset/17/breast-cancer
- Scikit-learn dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- Scikit-learn nested CV: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- Scikit-learn feature-selection pipeline: https://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection_pipeline.html
- Project 1 repository: https://github.com/Ehsan-Rezaeii/metaheuristics-lab
