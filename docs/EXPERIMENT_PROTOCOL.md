# Frozen experimental protocol

Registered before the full run, 2026-11-10. The original PROJECT_SPEC.md remains
unchanged as the pre-implementation contract. No scientific defaults are amended.

Research question: can binary search reduce features while preserving malignant
recall? Data are sklearn's bundled WDBC, UCI DOI 10.24432/C5DW2B. Source malignant=0
is explicitly remapped to positive=1. Dataset invariants: 569 x 30, malignant 212,
benign 357. Some predictors are zero despite the spec's positive-valued wording;
the loader permits nonnegative finite values. No identifiers are predictors.

Split: stratified 80/20, seed 2026. Only development data enter descriptive summaries,
optimization, or development plots. Outer stratified 3-fold CV uses seed 2027;
each outer training set has fresh stratified 3-fold inner CV, seed 2028. Optimizer
seeds are fold numbers 0,1,2. Indices in artifacts are original sklearn row indices.
No estimator sees outer validation data until prediction. No global scaling/filtering.

Fixed classifier: StandardScaler then LogisticRegression(C=1, solver=liblinear,
max_iter=2000, random_state=0), decision threshold 0.5. The filter comparator uses
SelectKBest(f_classif,k=10) before scaling. In the implementation its filter and
initial classifier fit use only training data; the resulting fixed mask is then
refit using the same classifier. Both actual classifier fits are counted.
AllFeatures uses all 30 predictors.

GA, GWO-S, GWO-V and binary RandomSearch are imported from the actual Project 1
distribution pinned at 6b4ef0fa3d72f608146f3dd7eabb84bf39fa896e. Population=12,
exactly 90 calls, including initialization and repeated masks, without caching.
Fitness minimized: 0.70*(1-mean recall)+0.30*(1-mean F1)+0.02*k/30.
An empty mask costs 2.0 and performs zero fits. Every nonempty objective call
performs 3 classifier fits. Failures propagate; there is no ignored error score.
Budgets measure calls, not equal elapsed time or equal convergence opportunity.
Independent local NumPy RNG instances use the same declared seed schedule for
paired method comparisons; they share no mutable generator state.

Full-development training-only repeats use seeds [0,1,2] and the same inner-CV
protocol and budget. Final choice per search method sorts by fitness, feature count,
seed, then mask lexicographically. Inner fitness is optimized and biased; outer CV
is the development estimate. Repeated seeds are not independent patient samples.
The baseline masks are also determined solely from development data.

The develop command writes freeze.json and its canonical JSON hash before final
holdout access. It includes all six masks, seeds, config hash, time, feature order,
and provenance. Final is a separate command with an exclusive access marker;
it refuses reruns in the same directory, including after partial failure. Each
frozen method is fitted on development and evaluated once. No revisions or winner
selection follow test inspection. A new independent split would be needed for
future confirmatory work; re-running this command is only computational reproduction.

Outputs: outer-fold rows, search-call masks/fitness/cumulative fits, optimizer history,
seed repeats, split/fold indices, versions and VCS dependency metadata, configs,
freeze and access timestamps, final metrics, summaries, figures, SHA256 manifest.
Confusion matrices use true rows/predicted columns [0 benign,1 malignant]. FN is
cell [1,0]. ROC-AUC uses the probability column whose classes_ value is 1.
Timing uses perf_counter wall seconds: search includes inner fits and optimizer;
evaluation fit and prediction are separate; outer total includes selection/refit.
No concurrency is used; BLAS thread counts are fixed to 1 in recorded commands.

Smoke is explicitly a pilot: 4 calls, population 3, one development seed; all
three outer folds remain present. It is not evidence for the full protocol.

Preflight end-to-end checks and the smoke command use deterministic synthetic data.
Data-invariant and optimization unit tests use development data from the bundled
WDBC dataset. An initial automated end-to-end test exercised the real-data final
code path with a one-call pilot; its metrics were neither displayed nor inspected
and did not inform protocol, seed or mask choices. Subsequent preflight checks use
synthetic data to avoid such access. The reported full study's holdout is evaluated
only after the full-study freeze. This procedural caveat is disclosed for provenance.
