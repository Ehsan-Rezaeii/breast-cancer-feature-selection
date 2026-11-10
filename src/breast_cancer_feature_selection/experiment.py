"""Explicit development/freeze/final/report stages with immutable output directories."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import PROJECT1_REVISION
from .data import load_data, split_indices
from .evaluation import classifier, malignant_scores, metrics
from .selection import METHODS, SEARCH_METHODS, inner_splits, search


@dataclass(frozen=True)
class Config:
    budget: int = 90
    population: int = 12
    seeds: tuple = (0, 1, 2)
    pilot: bool = False
    split_seed: int = 2026
    outer_seed: int = 2027
    inner_seed: int = 2028
    threshold: float = 0.5
    recall_weight: float = 0.70
    f1_weight: float = 0.30
    feature_weight: float = 0.02
    empty_penalty: float = 2.0

    def __post_init__(self):
        # These are recorded constants, not an undocumented tuning interface.
        fixed = (
            self.split_seed,
            self.outer_seed,
            self.inner_seed,
            self.threshold,
            self.recall_weight,
            self.f1_weight,
            self.feature_weight,
            self.empty_penalty,
        )
        if fixed != (2026, 2027, 2028, 0.5, 0.70, 0.30, 0.02, 2.0):
            raise ValueError("Protocol amendment required for changed scientific constants")
        if self.budget < 1 or self.population < 3 or not self.seeds:
            raise ValueError("Positive budget, population >=3, and seeds are required")
        if not self.pilot and (self.budget, self.population, self.seeds) != (90, 12, (0, 1, 2)):
            raise ValueError("Nondefault budgets/seeds must be explicitly marked pilot")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def git_revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def environment():
    direct = json.loads(metadata.distribution("metaheuristics-lab").read_text("direct_url.json"))
    assert direct["vcs_info"]["commit_id"] == PROJECT1_REVISION
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        packages={
            name: metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "scikit-learn",
                "pandas",
                "matplotlib",
                "metaheuristics-lab",
            )
        },
        project1=direct,
        git_revision=git_revision(),
        source_hashes={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("src").rglob("*.py"))
        },
    )


def baseline_mask(method, X, y):
    if method == "AllFeatures":
        return np.ones(X.shape[1], dtype=int).tolist(), 0
    # Supervised filter fits only the permitted training data.
    model = classifier(True).fit(X, y)
    return model.named_steps["select"].get_support().astype(int).tolist(), 1


def fit_score(mask, X, y, X_valid, y_valid):
    start = perf_counter()
    selected = np.asarray(mask, dtype=bool)
    model = classifier().fit(X[:, selected], y)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    scores = malignant_scores(model, X_valid[:, selected])
    return dict(
        **metrics(y_valid, scores),
        fit_seconds=fit_seconds,
        predict_seconds=perf_counter() - start,
        evaluation_fits=1,
        features=int(selected.sum()),
        feature_fraction=float(selected.mean()),
    )


def prepare_development(X, y, ids, config, output):
    """Receives only development arrays; cannot read test arrays or labels."""
    rows, traces, fold_manifest = [], [], []
    outer = StratifiedKFold(3, shuffle=True, random_state=config.outer_seed)
    for fold, (train, valid) in enumerate(outer.split(X, y)):
        fold_manifest.append(
            dict(
                fold=fold,
                optimizer_seed=fold,
                train_indices=ids[train].tolist(),
                valid_indices=ids[valid].tolist(),
                inner=[
                    dict(train=ids[train][a].tolist(), valid=ids[train][b].tolist())
                    for a, b in inner_splits(X[train], y[train])
                ],
            )
        )
        for method in METHODS:
            start = perf_counter()
            if method in SEARCH_METHODS:
                run = search(method, X[train], y[train], fold, config.budget, config.population)
                traces.append(dict(stage="outer", fold=fold, method=method, **run))
                mask, baseline_fits = run["mask"], 0
            else:
                mask, baseline_fits = baseline_mask(method, X[train], y[train])
                run = dict(fitness=None, objective_calls=0, search_fits=0, search_seconds=0.0)
            row = dict(
                stage="outer",
                fold=fold,
                seed=fold,
                method=method,
                mask=json.dumps(mask),
                baseline_selection_fits=baseline_fits,
                **{
                    k: run[k]
                    for k in ("fitness", "objective_calls", "search_fits", "search_seconds")
                },
                **fit_score(mask, X[train], y[train], X[valid], y[valid]),
            )
            row["total_seconds"] = perf_counter() - start
            rows.append(row)
            print(f"outer fold={fold} {method}: recall={row['recall']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(output / "development_rows.csv", index=False)
    write_json(output / "folds.json", fold_manifest)
    final, repeats = {}, []
    for method in METHODS:
        if method in SEARCH_METHODS:
            runs = []
            for seed in config.seeds:
                run = search(method, X, y, seed, config.budget, config.population)
                traces.append(dict(stage="development_repeat", method=method, **run))
                repeats.append(
                    dict(
                        method=method,
                        **{k: v for k, v in run.items() if k not in ("trace", "history")},
                    )
                )
                runs.append(run)
            # Predeclared lexicographic tie-break: fitness, feature count, seed, mask.
            best = min(runs, key=lambda r: (r["fitness"], sum(r["mask"]), r["seed"], r["mask"]))
            final[method] = {k: v for k, v in best.items() if k not in ("trace", "history")}
        else:
            mask, fits = baseline_mask(method, X, y)
            final[method] = dict(
                mask=mask, seed=None, objective_calls=0, search_fits=0, baseline_selection_fits=fits
            )
    pd.DataFrame(repeats).to_json(output / "seed_repeats.json", orient="records", indent=2)
    write_json(output / "search_traces.json", traces)
    return final


def develop(output, config, data=None):
    output.mkdir(parents=True, exist_ok=False)
    X, y, names = load_data() if data is None else data
    dev, test = split_indices(y)
    write_json(output / "config.json", asdict(config))
    write_json(output / "environment.json", environment())
    write_json(output / "split.json", dict(development=dev.tolist(), test=test.tolist()))
    # Descriptions and plots during development use development data only.
    pd.DataFrame(X[dev], columns=names).describe().T.to_csv(output / "feature_summary.csv")
    write_json(
        output / "class_summary.json",
        dict(
            stage="development", benign=int((y[dev] == 0).sum()), malignant=int((y[dev] == 1).sum())
        ),
    )
    final = prepare_development(X[dev], y[dev], dev, config, output)
    freeze = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        config_hash=digest(asdict(config)),
        methods=final,
        metric_names=["recall", "f1", "precision", "roc_auc", "tn", "fp", "fn", "tp"],
        development_indices_hash=digest(dev.tolist()),
        split_hash=digest(dict(development=dev.tolist(), test=test.tolist())),
        feature_names=names,
        environment_hash=digest(json.loads((output / "environment.json").read_text())),
        full_development_inner=[
            dict(train=dev[a].tolist(), valid=dev[b].tolist())
            for a, b in inner_splits(X[dev], y[dev])
        ],
    )
    write_json(output / "freeze.json", freeze)
    (output / "freeze.sha256").write_text(digest(freeze) + "\n")
    print(f"FROZEN {output}: {digest(freeze)}", flush=True)


def evaluate_final(output, data=None):
    freeze = json.loads((output / "freeze.json").read_text())
    assert digest(freeze) == (output / "freeze.sha256").read_text().strip()
    assert freeze["config_hash"] == digest(json.loads((output / "config.json").read_text()))
    recorded_environment = json.loads((output / "environment.json").read_text())
    assert freeze["environment_hash"] == digest(recorded_environment)
    current_environment = environment()
    for key in ("packages", "source_hashes", "project1"):
        assert current_environment[key] == recorded_environment[key], f"Changed {key} since freeze"
    split = json.loads((output / "split.json").read_text())
    assert digest(split) == freeze["split_hash"]
    # Exclusive marker refuses accidental repeated holdout access, even after a partial failure.
    with (output / "holdout_started.json").open("x") as stream:
        json.dump(
            dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), freeze_hash=digest(freeze)),
            stream,
        )
    X, y, _ = load_data() if data is None else data
    split = json.loads((output / "split.json").read_text())
    dev, test = np.array(split["development"]), np.array(split["test"])
    assert digest(dev.tolist()) == freeze["development_indices_hash"]
    rows = []
    for method, run in freeze["methods"].items():
        rows.append(
            dict(
                stage="final_holdout",
                method=method,
                seed=run["seed"],
                mask=json.dumps(run["mask"]),
                **fit_score(run["mask"], X[dev], y[dev], X[test], y[test]),
            )
        )
    pd.DataFrame(rows).to_csv(output / "final_rows.csv", index=False)
    write_json(
        output / "holdout_completed.json",
        dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), freeze_hash=digest(freeze)),
    )
    print(pd.DataFrame(rows).drop(columns="mask").to_string(index=False), flush=True)


def report(output):
    rows = pd.read_csv(output / "development_rows.csv")
    summary = rows.groupby("method").agg(
        {
            name: ["mean", "std"]
            for name in (
                "recall",
                "f1",
                "precision",
                "roc_auc",
                "features",
                "fn",
                "search_seconds",
                "total_seconds",
            )
        }
    )
    summary.to_csv(output / "development_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in rows.groupby("method"):
        ax.scatter(group.features, group.recall, label=method)
    ax.set(
        xlabel="Selected features",
        ylabel="Malignant recall (outer development CV)",
        title="Fresh selection within each outer training fold",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "development_recall_features.png", dpi=150)
    plt.close(fig)
    final = pd.read_csv(output / "final_rows.csv")
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, (_, row) in zip(axes.flat, final.iterrows(), strict=True):
        cm = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        ax.imshow(cm, cmap="Blues")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center")
        ax.set(
            title=row.method,
            xticks=[0, 1],
            yticks=[0, 1],
            xlabel="Predicted (0 benign, 1 malignant)",
            ylabel="True",
        )
    fig.tight_layout()
    fig.savefig(output / "final_confusion_matrices.png", dpi=150)
    plt.close(fig)
    checksums = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir())
        if p.is_file() and p.name != "checksums.json"
    }
    write_json(output / "checksums.json", checksums)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["develop", "final", "report", "smoke"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = None
    if args.stage == "smoke":
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=569, n_features=30, n_informative=10, random_state=17)
        data = (X, y, [f"synthetic_{i}" for i in range(30)])
    if args.stage in ("develop", "smoke"):
        develop(
            args.output,
            Config(budget=4, population=3, seeds=(0,), pilot=True)
            if args.stage == "smoke"
            else Config(),
            data=data,
        )
    if args.stage in ("final", "smoke"):
        evaluate_final(args.output, data=data)
    if args.stage in ("report", "smoke"):
        report(args.output)


if __name__ == "__main__":
    main()
