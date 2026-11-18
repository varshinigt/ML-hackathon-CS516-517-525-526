"""Model zoo for candidate classification (one entry per experimented model).

Every model outputs P(candidate is the reference arc | features). Models that are expensive
in the number of training rows (KNN, SVM, MLP) are fitted on a stratified, trip-respecting
row subsample; the subsample size is part of the recorded configuration.
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42


def _lgbm(**kw):
    from lightgbm import LGBMClassifier
    base = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=-1)
    base.update(kw)
    return LGBMClassifier(**base)


# name -> (family, factory, max_train_rows or None, config description)
MODEL_ZOO = {
    "LogisticRegression": ("Linear / probabilistic",
        lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        None, "StandardScaler + L2 logistic regression, C=1.0, lbfgs"),
    "GaussianNB": ("Linear / probabilistic",
        lambda: GaussianNB(var_smoothing=1e-9), None, "Gaussian NB, var_smoothing=1e-9"),
    "KNN": ("Instance-based",
        lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
        60_000, "StandardScaler + distance-weighted KNN, k=25, fit on 60k-row subsample"),
    "DecisionTree": ("Tree-based",
        lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=SEED),
        None, "max_depth=12, min_samples_leaf=50"),
    "RandomForest": ("Ensemble (bagging)",
        lambda: RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=20, n_jobs=-1, random_state=SEED),
        None, "200 trees, max_depth=18, min_samples_leaf=20"),
    "ExtraTrees": ("Ensemble (bagging)",
        lambda: ExtraTreesClassifier(n_estimators=200, max_depth=18, min_samples_leaf=20, n_jobs=-1, random_state=SEED),
        None, "200 trees, max_depth=18, min_samples_leaf=20"),
    "HistGradientBoosting": ("Boosting",
        lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=SEED),
        None, "300 iters, lr=0.08, 31 leaves"),
    "LightGBM": ("Boosting", _lgbm, None, "400 trees, lr=0.05, 63 leaves, bagging 0.8/0.8"),
    "LinearSVM": ("Kernel-based",
        lambda: make_pipeline(StandardScaler(), CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=5000), cv=3, method="sigmoid")),
        150_000, "StandardScaler + LinearSVC C=0.5, Platt-calibrated (3-fold), 150k-row subsample"),
    "MLP": ("Neural network",
        lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512,
                                                              learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                                                              n_iter_no_change=6, random_state=SEED)),
        300_000, "StandardScaler + MLP 64-32 ReLU, Adam lr=1e-3, early stopping, 300k-row subsample"),
}


def make_tuned(name: str, params: dict):
    """Factory for the selected model with Bayesian-optimised hyperparameters."""
    if name == "LightGBM":
        return lambda: _lgbm(**params)
    if name == "HistGradientBoosting":
        return lambda: HistGradientBoostingClassifier(**params, random_state=SEED)
    if name == "RandomForest":
        return lambda: RandomForestClassifier(**params, n_jobs=-1, random_state=SEED)
    if name == "ExtraTrees":
        return lambda: ExtraTreesClassifier(**params, n_jobs=-1, random_state=SEED)
    raise KeyError(name)
