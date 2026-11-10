"""Application fitness adapter; optimizer implementations belong to Project 1."""

from time import perf_counter

import numpy as np
from metaheuristics_lab import BinaryGA, BinaryGWO, Problem, RandomSearch
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

from .evaluation import classifier, malignant_scores

SEARCH_METHODS = ("BinaryGA", "BinaryGWO-S", "BinaryGWO-V", "RandomSearch(binary)")
METHODS = ("AllFeatures", "SelectKBest(k=10)", *SEARCH_METHODS)


def inner_splits(X, y):
    return list(StratifiedKFold(3, shuffle=True, random_state=2028).split(X, y))


class Fitness:
    def __init__(self, X, y, splits):
        self.X, self.y, self.splits = X, y, splits
        self.calls, self.fits, self.trace = 0, 0, []

    def __call__(self, candidate):
        bits = np.asarray(candidate)
        if bits.shape != (self.X.shape[1],) or not np.isin(bits, [0, 1]).all():
            raise ValueError("mask must have one binary bit per feature")
        mask = bits.astype(bool)
        self.calls += 1
        recall, f1 = [], []
        value = 2.0
        if mask.any():
            for train, valid in self.splits:
                model = classifier().fit(self.X[train][:, mask], self.y[train])
                self.fits += 1
                pred = (malignant_scores(model, self.X[valid][:, mask]) >= 0.5).astype(int)
                recall.append(recall_score(self.y[valid], pred, zero_division=0))
                f1.append(f1_score(self.y[valid], pred, zero_division=0))
            value = 0.70 * (1 - np.mean(recall)) + 0.30 * (1 - np.mean(f1))
            value += 0.02 * mask.mean()
        self.trace.append(
            dict(
                call=self.calls,
                mask=bits.astype(int).tolist(),
                fitness=float(value),
                cumulative_fits=self.fits,
            )
        )
        return float(value)


def search(method, X, y, seed, budget=90, population=12):
    optimizers = {
        "BinaryGA": lambda: BinaryGA(population_size=population),
        "BinaryGWO-S": lambda: BinaryGWO(population_size=population, transfer="s"),
        "BinaryGWO-V": lambda: BinaryGWO(population_size=population, transfer="v"),
        "RandomSearch(binary)": lambda: RandomSearch(population_size=population, domain="binary"),
    }
    fitness = Fitness(X, y, inner_splits(X, y))
    start = perf_counter()
    result = optimizers[method]().optimize(
        Problem.binary(fitness, dimension=X.shape[1]), max_evaluations=budget, seed=seed
    )
    assert result.n_evaluations == fitness.calls == budget
    mask = result.best_solution.astype(bool)
    assert mask.any()
    return dict(
        mask=mask.astype(int).tolist(),
        fitness=result.best_fitness,
        seed=seed,
        objective_calls=fitness.calls,
        search_fits=fitness.fits,
        search_seconds=perf_counter() - start,
        trace=fitness.trace,
        history=result.history.tolist(),
    )
