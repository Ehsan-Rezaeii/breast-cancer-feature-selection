"""Run a disposable synthetic pilot; never access the scientific holdout."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from sklearn.datasets import make_classification

from breast_cancer_feature_selection.experiment import Config, develop, evaluate_final, report


def synthetic_data():
    X, y = make_classification(n_samples=569, n_features=30, n_informative=10, random_state=17)
    return X, y, [f"synthetic_{i}" for i in range(30)]


if __name__ == "__main__":
    with (
        tempfile.TemporaryDirectory() as directory,
        patch("breast_cancer_feature_selection.experiment.load_data", synthetic_data),
    ):
        output = Path(directory) / "pilot"
        develop(output, Config(budget=4, population=3, seeds=(0,), pilot=True))
        evaluate_final(output)
        report(output)
