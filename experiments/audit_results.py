"""Audit saved artifacts without accessing or refitting on the holdout."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from breast_cancer_feature_selection.experiment import digest
from breast_cancer_feature_selection.selection import SEARCH_METHODS


def audit(output: Path):
    read = lambda name: json.loads((output / name).read_text())  # noqa: E731
    for name, expected in read("checksums.json").items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected, name
    for name, expected in read("environment.json")["source_hashes"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    config, freeze, split = read("config.json"), read("freeze.json"), read("split.json")
    assert digest(freeze) == (output / "freeze.sha256").read_text().strip()
    assert digest(config) == freeze["config_hash"]
    assert digest(split) == freeze["split_hash"]
    dev, test = set(split["development"]), set(split["test"])
    assert not dev & test and len(dev | test) == 569
    folds = read("folds.json")
    for fold in folds:
        train, valid = set(fold["train_indices"]), set(fold["valid_indices"])
        assert train | valid == dev and not train & valid
        for inner in fold["inner"]:
            a, b = set(inner["train"]), set(inner["valid"])
            assert a | b == train and not a & b
    traces = read("search_traces.json")
    assert len(traces) == 4 * (3 + len(config["seeds"]))
    for run in traces:
        assert run["objective_calls"] == len(run["trace"]) == config["budget"]
        fits = 0
        for call, point in enumerate(run["trace"], 1):
            fits += 3 if any(point["mask"]) else 0
            assert point["call"] == call and point["cumulative_fits"] == fits
        assert run["search_fits"] == fits
        assert all(
            len(point["mask"]) == 30 and set(point["mask"]) <= {0, 1} for point in run["trace"]
        )
        assert any(
            point["mask"] == run["mask"] and point["fitness"] == run["fitness"]
            for point in run["trace"]
        )
        np.testing.assert_allclose(
            run["history"], np.minimum.accumulate([point["fitness"] for point in run["trace"]])
        )
        assert run["fitness"] == min(point["fitness"] for point in run["trace"])
    outer = pd.read_csv(output / "development_rows.csv")
    final = pd.read_csv(output / "final_rows.csv")
    assert len(outer) == 18 and len(final) == 6
    for frame in (outer, final):
        np.testing.assert_allclose(frame.recall, frame.tp / (frame.tp + frame.fn))
        np.testing.assert_allclose(frame.precision, frame.tp / (frame.tp + frame.fp))
        np.testing.assert_allclose(frame.f1, 2 * frame.tp / (2 * frame.tp + frame.fp + frame.fn))
        for _, row in frame.iterrows():
            assert sum(json.loads(row["mask"])) == row.features
    for method in SEARCH_METHODS:
        candidates = [
            run
            for run in traces
            if run["stage"] == "development_repeat" and run["method"] == method
        ]
        best = min(candidates, key=lambda r: (r["fitness"], sum(r["mask"]), r["seed"], r["mask"]))
        assert freeze["methods"][method]["mask"] == best["mask"]
        row = final.loc[final.method == method].iloc[0]
        assert json.loads(row["mask"]) == best["mask"]
    saved = pd.read_csv(output / "development_summary.csv", header=[0, 1], index_col=0)
    for name, statistic in saved.columns:
        expected = outer.groupby("method")[name].agg(statistic)
        np.testing.assert_allclose(saved[(name, statistic)].values, expected.values)
    assert freeze["timestamp_utc"] < read("holdout_started.json")["timestamp_utc"]
    assert (
        read("holdout_started.json")["timestamp_utc"]
        < read("holdout_completed.json")["timestamp_utc"]
    )
    cost = []
    for method in final.method:
        runs = [r for r in traces if r["method"] == method]
        cost.append(
            dict(
                method=method,
                objective_calls=sum(r["objective_calls"] for r in runs),
                inner_fits=sum(r["search_fits"] for r in runs),
                baseline_selection_fits=4 if method == "SelectKBest(k=10)" else 0,
                evaluation_fits=4,
                search_seconds=sum(r["search_seconds"] for r in runs),
            )
        )
    pd.DataFrame(cost).to_csv(output / "cost_summary.csv", index=False)
    print(
        "PASS: hashes, splits, fold nesting, budgets, fits, metrics, freeze choices and timestamps"
    )
    print(pd.DataFrame(cost).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)
