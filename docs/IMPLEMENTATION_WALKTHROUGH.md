# Implementation walkthrough

This English guide is intended for a separate learning session after implementation.
Start with the research question in PROJECT_SPEC.md and the executable constants
in experiment.Config. The code deliberately fixes classifier and fitness settings;
there is no post-test tuning interface. Read RESULTS.md before drawing conclusions.

## 1. Package and actual dependency

pyproject.toml defines an installable src-layout package and bcfs-study entry point.
Its direct VCS requirement installs the genuine Project 1 distribution at immutable
commit 6b4ef0fa3d72f608146f3dd7eabb84bf39fa896e. selection.py imports Problem,
BinaryGA, BinaryGWO and RandomSearch from metaheuristics_lab. It implements only
the application objective and dispatch, never optimizer updates. environment()
checks installed PEP 610 direct_url.json, records package versions and hashes all
study source modules. Tests inspect classes' defining modules and actual revision.
No PyPI publication of Project 1 is assumed.

## 2. Data, label direction and allowed information

data.load_data verifies dataset shape, target names, class counts, finiteness and
nonnegative predictors. Mapping `(target == 0).astype(int)` makes malignancy 1.
A reversed label convention could yield reassuring but irrelevant benign recall;
the tests explicitly prevent this mistake. No row indices are model predictors.
Indices serve only as reproducibility/audit identifiers.

split_indices uses the fixed stratified seed to create 455 development and 114
holdout rows. Stratification necessarily uses labels to allocate rows. That is
split construction, not model fitting. Only development arrays are passed to
prepare_development. Development summaries never describe the scientific holdout.
The loader's source-integrity class-count assertion is a predefined dataset check.

## 3. Estimators and metric semantics

evaluation.classifier builds the fixed scaler/logistic pipeline. For SelectKBest,
the filter is inside a pipeline and precedes scaling. baseline_mask fits this
pipeline on allowed training data and extracts the support; fit_score refits the
same scaler/logistic estimator with that fixed support. This extra classifier fit
is computationally redundant but explicitly counted, and does not change leakage
boundaries. AllFeatures uses an all-one mask without a selection fit.

malignant_scores finds `classes_ == 1`; it does not assume probability-column
ordering. metrics applies the fixed >=0.5 threshold, reports recall/F1/precision
for positive=1 and uses continuous class-1 scores for ROC-AUC. The matrix convention
is [[TN,FP],[FN,TP]] with true rows/predicted columns [benign,malignant]. The reversed
probability-column test is deliberately adversarial.

## 4. Inner fitness and resource accounting

Fitness owns only its input training arrays and inner index pairs. Each nonempty
mask causes three fresh pipeline fits, each on an inner training subset, followed
by prediction on its validation subset. Mean fold recall and F1 enter the
predeclared weighted loss; feature count adds 0.02*k/30. Empty masks return 2.0
without fitting. Bits must be binary, with 1 meaning selected.

Every invocation increments calls. Repeated masks are intentionally recomputed,
so objective_calls and search_fits have transparent meanings. Each call saves
mask, fitness and cumulative fits; Project 1 supplies best-so-far history.
search asserts the actual package consumed exactly the requested budget, even
when the last population is partial. Equal calls do not imply equal wall time,
independent trajectories, or equal difficulty for each algorithm.

## 5. Nested CV and development seed repeats

prepare_development creates three outer folds. For each method/fold, its selector
receives only the outer training arrays. Search constructs fresh inner folds with
the same fixed seed so all search methods within an outer fold see the same
indices. The final outer-training classifier predicts outer validation only after
selection. Outer validation labels never participate in fitness. folds.json maps
all local indices back to original row indices for independent inspection.

The full-development repeats then search with seeds 0,1,2. These measure optimizer
variability on the same patients; they are not independent validation cohorts.
The deterministic selection rule is fitness, feature count, seed, mask. Inner
fitness was optimized and must not be reported as unbiased generalization.

The holdout-perturbation regression test changes held-out features and labels,
runs the development boundary again and compares the chosen masks and seeds.
The nested-boundary test instruments the selector and checks exact outer-training
rows. Another test captures each inner fit's arrays and compares them to the
expected inner-training arrays. These are behavioral tests of leakage boundaries,
not simply assertions that train and test indices differ.

## 6. Freeze and final stage

develop saves a timestamped manifest with every method's mask, source/dependency
provenance, seed, fitness, config hash, split hash and intended metrics. It then
writes its canonical JSON SHA256. The full-budget protocol and source code were
committed before this command. evaluate_final verifies freeze/config/split and
source/dependency integrity, creates an exclusive holdout-start marker, fits each
frozen mask on development, and evaluates each once. A partial failure leaves the
marker in place, requiring investigation rather than silently trying again.

The marker is a practical guard against mistakes, not a security boundary: someone
can delete files or create new output directories. Scientific discipline and the
committed chronology remain necessary. Preflight synthetic smoke is clearly a
pilot. See the procedural caveat in EXPERIMENT_PROTOCOL.md about an initial
automated real-data test code-path exercise with no inspected performance outputs.

## 7. Reports, provenance and testing

report reads raw rows to aggregate outer means/sample standard deviations and draw
outer-development recall versus features plus six final confusion matrices.
It does not refit models. The artifact checksum manifest excludes only itself.
audit_results.py verifies files, partitions, inner/outer nesting, call traces,
fit counts, derived confusion-matrix metrics, final seed choices, freeze chronology
and summary recall. It writes an explicit total cost summary from all recorded runs.
It cannot independently reconstruct ROC-AUC without patient-level predictions;
that computation is covered by score semantics tests. No patient-level predictions
or raw predictor records are committed.

CI installs the direct dependency, checks lint/format and tests, runs a synthetic
smoke, builds sdist/wheel, installs the wheel in a clean environment and imports
from outside the checkout. The long full-budget study is opt-in. Developer tests
must not be interpreted as new scientific holdout measurements. Timing and fonts
can vary between machines even when masks and scores reproduce.

## 8. Suggested later study order

Read data.py, evaluation.py and their tests first. Then manually trace one fitness
call through three folds. Read selection.py alongside Project 1's public API.
Finally follow experiment.py from outer folds through development repeats to the
freeze and final stage. Compare raw CSVs with RESULTS.md and run the artifact audit.
The central distinction is between an optimized inner score, an outer estimate of
the selection procedure, and a single holdout estimate for frozen final masks.
