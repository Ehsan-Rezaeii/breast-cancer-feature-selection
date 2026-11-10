"""Verified bundled UCI data and predeclared index split."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data():
    data = load_breast_cancer()
    assert list(data.target_names) == ["malignant", "benign"]
    X = np.asarray(data.data, dtype=float)
    y = (data.target == 0).astype(int)
    assert X.shape == (569, 30) and np.isfinite(X).all()
    assert np.bincount(y).tolist() == [357, 212]
    # Some valid predictors are zero; reject negatives, not zeros.
    assert (X >= 0).all()
    return X, y, data.feature_names.tolist()


def split_indices(y):
    return train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=2026)
