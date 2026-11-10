"""Fold-local estimators and malignant-positive metrics."""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def classifier(filter_features=False):
    steps = [("select", SelectKBest(f_classif, k=10))] if filter_features else []
    return Pipeline(
        steps
        + [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=0)),
        ]
    )


def malignant_scores(model, X):
    column = np.flatnonzero(model.classes_ == 1).item()
    return model.predict_proba(X)[:, column]


def metrics(y, scores):
    predicted = (np.asarray(scores) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(
        recall=recall_score(y, predicted, pos_label=1, zero_division=0),
        f1=f1_score(y, predicted, pos_label=1, zero_division=0),
        precision=precision_score(y, predicted, pos_label=1, zero_division=0),
        roc_auc=roc_auc_score(y, scores),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        malignant_support=int(tp + fn),
    )
