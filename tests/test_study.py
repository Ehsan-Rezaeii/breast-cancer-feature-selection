"""Scientific invariants, real-dependency integration, and orchestration regression tests."""

import json

import numpy as np
import pytest
from metaheuristics_lab import BinaryGA, BinaryGWO, Problem, RandomSearch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score

from breast_cancer_feature_selection import PROJECT1_REVISION
from breast_cancer_feature_selection.data import load_data, split_indices
from breast_cancer_feature_selection.evaluation import classifier, malignant_scores, metrics
from breast_cancer_feature_selection.experiment import (
    Config,
    develop,
    environment,
    evaluate_final,
    prepare_development,
    report,
)
from breast_cancer_feature_selection.selection import SEARCH_METHODS, Fitness, inner_splits, search


def test_dataset_and_split():
    X, y, names = load_data()
    assert X.shape == (569, 30) and len(names) == 30
    assert np.array_equal(y, load_breast_cancer().target == 0)
    assert np.bincount(y).tolist() == [357, 212]
    dev, test = split_indices(y)
    assert len(dev) == 455 and len(test) == 114
    assert not set(dev) & set(test)
    assert len(set(dev) | set(test)) == 569
    assert all(np.array_equal(a, b) for a, b in zip((dev, test), split_indices(y), strict=True))


def test_metrics_positive_and_probability_column():
    y = np.array([1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.7, 0.1, 0.8, 0.2])
    row = metrics(y, scores)
    assert row["recall"] == pytest.approx(2 / 3)
    assert row["precision"] == pytest.approx(2 / 3)
    assert row["f1"] == pytest.approx(2 / 3)
    assert (row["tn"], row["fp"], row["fn"], row["tp"]) == (1, 1, 1, 2)
    assert row["roc_auc"] == roc_auc_score(y, scores)

    class ReversedColumns:
        classes_ = np.array([1, 0])

        def predict_proba(self, X):
            return np.column_stack([scores, 1 - scores])

    assert np.array_equal(malignant_scores(ReversedColumns(), None), scores)
    assert metrics([0, 1], [0.5, 0.5])["recall"] == 1


def test_fold_local_scaler_and_filter():
    X, y, _ = load_data()
    train, valid = inner_splits(X, y)[0]
    model = classifier(True).fit(X[train], y[train])
    selected = model.named_steps["select"].get_support()
    assert selected.sum() == 10
    np.testing.assert_allclose(model.named_steps["scale"].mean_, X[train][:, selected].mean(0))
    assert model.named_steps["scale"].n_samples_seen_ == len(train)
    assert len(valid) > 0


def test_fitness_empty_mask_repeats_and_train_only(monkeypatch):
    import breast_cancer_feature_selection.selection as selection

    X, y, _ = load_data()
    X, y = X[:120], y[:120]
    splits = inner_splits(X, y)
    original = classifier
    seen = []

    def instrumented():
        pipeline = original()
        real_fit = pipeline.fit

        def fit(A, b):
            seen.append((A.copy(), b.copy()))
            return real_fit(A, b)

        pipeline.fit = fit
        return pipeline

    monkeypatch.setattr(selection, "classifier", instrumented)
    fitness = Fitness(X, y, splits)
    assert fitness(np.zeros(30)) == 2.0 and fitness.fits == 0
    mask = np.zeros(30)
    mask[[0, 5]] = 1
    value = fitness(mask)
    assert value == fitness(mask)
    assert fitness.calls == 3 and fitness.fits == 6
    for (A, b), (train, valid) in zip(seen[:3], splits, strict=True):
        np.testing.assert_array_equal(A, X[train][:, mask.astype(bool)])
        np.testing.assert_array_equal(b, y[train])
        assert not set(train) & set(valid)
    with pytest.raises(ValueError):
        fitness(np.full(30, 0.2))


@pytest.mark.parametrize("method", SEARCH_METHODS)
def test_real_optimizer_budget_and_determinism(method):
    X, y, _ = load_data()
    dev, _ = split_indices(y)
    a = search(method, X[dev], y[dev], seed=4, budget=7, population=3)
    b = search(method, X[dev], y[dev], seed=4, budget=7, population=3)
    assert a["objective_calls"] == 7
    assert a["search_fits"] == 21
    for key in ("mask", "fitness", "trace", "history"):
        assert a[key] == b[key]
    assert sum(a["mask"]) > 0


def test_dependency_is_installed_distribution():
    assert environment()["project1"]["vcs_info"]["commit_id"] == PROJECT1_REVISION
    for cls in (Problem, BinaryGA, BinaryGWO, RandomSearch):
        assert cls.__module__.startswith("metaheuristics_lab.")


def test_nested_selection_boundaries_and_holdout_perturbation(tmp_path, monkeypatch):
    import breast_cancer_feature_selection.experiment as experiment

    X, y, _ = load_data()
    dev, test = split_indices(y)
    observed = []
    actual_search = search

    def inspected(method, A, b, *args):
        # Every row in a selector input must be a development row.
        allowed = {tuple(row) for row in X[dev]}
        assert all(tuple(row) in allowed for row in A)
        observed.append(A.copy())
        return actual_search(method, A, b, *args)

    monkeypatch.setattr(experiment, "search", inspected)
    config = Config(budget=1, population=3, seeds=(0,), pilot=True)
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    first = prepare_development(X[dev], y[dev], dev, config, one)
    changed_X, changed_y = X.copy(), y.copy()
    changed_y[test] = 1 - y[test]
    changed_X[test] = -999999
    second = prepare_development(changed_X[dev], changed_y[dev], dev, config, two)
    for method in first:
        assert first[method]["mask"] == second[method]["mask"]
        assert first[method]["seed"] == second[method]["seed"]
    folds = json.loads((one / "folds.json").read_text())
    for fold, A in zip(folds, observed[:12:4], strict=True):
        np.testing.assert_array_equal(A, X[fold["train_indices"]])
        assert not set(fold["train_indices"]) & set(fold["valid_indices"])
        for inner in fold["inner"]:
            assert set(inner["train"]) | set(inner["valid"]) == set(fold["train_indices"])
            assert not set(inner["train"]) & set(inner["valid"])


def test_end_to_end_freeze_and_one_time_holdout(tmp_path, monkeypatch):
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=569, n_features=30, random_state=17)
    monkeypatch.setattr(
        "breast_cancer_feature_selection.experiment.load_data",
        lambda: (X, y, [str(i) for i in range(30)]),
    )
    output = tmp_path / "pilot"
    develop(output, Config(budget=1, population=3, seeds=(0,), pilot=True))
    assert not (output / "final_rows.csv").exists()
    freeze = json.loads((output / "freeze.json").read_text())
    assert len(freeze["methods"]) == 6
    evaluate_final(output)
    with pytest.raises(FileExistsError):
        evaluate_final(output)
    report(output)
    assert (output / "checksums.json").exists()
    assert (output / "final_confusion_matrices.png").stat().st_size > 0


def test_fixed_protocol_requires_explicit_amendment():
    with pytest.raises(ValueError, match="amendment"):
        Config(feature_weight=0.5)
    with pytest.raises(ValueError, match="pilot"):
        Config(budget=1)


def test_freeze_tampering_fails_before_holdout(tmp_path, monkeypatch):
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=569, n_features=30, random_state=17)
    monkeypatch.setattr(
        "breast_cancer_feature_selection.experiment.load_data",
        lambda: (X, y, [str(i) for i in range(30)]),
    )
    output = tmp_path / "pilot"
    develop(output, Config(budget=1, population=3, seeds=(0,), pilot=True))
    split = json.loads((output / "split.json").read_text())
    split["test"][0] = split["development"][0]
    (output / "split.json").write_text(json.dumps(split))
    with pytest.raises(AssertionError):
        evaluate_final(output)
    assert not (output / "holdout_started.json").exists()
