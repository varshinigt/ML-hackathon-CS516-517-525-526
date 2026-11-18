"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected model.

Objective : mean observation-level Top-1 accuracy over 3-fold GroupKFold on the 80 training
            trips (the same quantity the route decoder consumes: which candidate wins per fix).
            Macro F1 and log loss are recorded for every trial as well.
Only training trips are used for the search; the tuned model is then refit on all 80 training
trips and compared with the untuned model on the 10 student-testing trips.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from pathsnap.experiment import fit_model, load_partition, obs_top1, predict, route_scores, score_block  # noqa: E402
from pathsnap.io import ROOT  # noqa: E402
from pathsnap.models import MODEL_ZOO, SEED  # noqa: E402

RES = ROOT / "results"


def space_lightgbm(trial):
    from lightgbm import LGBMClassifier
    prm = dict(
        n_estimators=trial.suggest_int("n_estimators", 150, 900, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 400, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    )
    return prm, lambda: LGBMClassifier(**prm, subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=-1)


def space_hgb(trial):
    from sklearn.ensemble import HistGradientBoostingClassifier
    prm = dict(
        max_iter=trial.suggest_int("max_iter", 100, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 400, log=True),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
    )
    return prm, lambda: HistGradientBoostingClassifier(**prm, random_state=SEED)


def space_rf(trial):
    from sklearn.ensemble import RandomForestClassifier
    prm = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
        max_depth=trial.suggest_int("max_depth", 8, 30),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 100, log=True),
        max_features=trial.suggest_float("max_features", 0.2, 1.0),
    )
    return prm, lambda: RandomForestClassifier(**prm, n_jobs=-1, random_state=SEED)


def space_et(trial):
    from sklearn.ensemble import ExtraTreesClassifier
    prm, _ = space_rf(trial)
    return prm, lambda: ExtraTreesClassifier(**prm, n_jobs=-1, random_state=SEED)


SPACES = {"LightGBM": space_lightgbm, "HistGradientBoosting": space_hgb, "RandomForest": space_rf,
          "ExtraTrees": space_et}


def main(model_name, n_trials):
    tr_c, tr_p = load_partition("train")
    st_c, st_p = load_partition("student_test")
    folds = list(GroupKFold(n_splits=3).split(tr_c, groups=tr_c["trip_id"]))
    space = SPACES[model_name]

    def objective(trial):
        prm, factory = space(trial)
        tops, f1s, lls, t_all = [], [], [], 0.0
        for i_tr, i_va in folds:
            m, t_fit, _ = fit_model(factory, tr_c.iloc[i_tr])
            va = tr_c.iloc[i_va]
            p, _ = predict(m, va)
            s = score_block(va, tr_p[tr_p["trip_id"].isin(set(va["trip_id"]))], p)
            tops.append(s["top1_acc"]); f1s.append(s["macro_f1"]); lls.append(s["log_loss"]); t_all += t_fit
        trial.set_user_attr("cv_macro_f1", float(np.mean(f1s)))
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("cv_top1_std", float(np.std(tops)))
        trial.set_user_attr("fit_seconds", t_all)
        print(f"trial {trial.number}: top1={np.mean(tops):.4f} macroF1={np.mean(f1s):.4f} {prm}", flush=True)
        return float(np.mean(tops))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    search_s = time.time() - t0
    trials = study.trials_dataframe()
    trials.to_csv(RES / f"bayes_opt_trials_{model_name}.csv", index=False)

    # refit best + untuned baseline on all training trips, evaluate on student test
    best_prm, best_factory = space(optuna.trial.FixedTrial(study.best_params))
    rows = []
    for tag, factory in (("untuned", MODEL_ZOO[model_name][1]), ("tuned", best_factory)):
        m, t_fit, _ = fit_model(factory, tr_c)
        p, t_inf = predict(m, st_c)
        s = score_block(st_c, st_p, p)
        rs = route_scores(obs_top1(st_c, st_p, p))
        cv_f1 = []
        cv_top = []
        for i_tr, i_va in folds:
            mm, _, _ = fit_model(factory, tr_c.iloc[i_tr])
            va = tr_c.iloc[i_va]
            pp, _ = predict(mm, va)
            ss = score_block(va, tr_p[tr_p["trip_id"].isin(set(va["trip_id"]))], pp)
            cv_f1.append(ss["macro_f1"]); cv_top.append(ss["top1_acc"])
        rows.append({"variant": tag, "cv_macro_f1": np.mean(cv_f1), "cv_top1": np.mean(cv_top),
                     "st_top1_acc": s["top1_acc"], "st_macro_f1": s["macro_f1"], "st_log_loss": s["log_loss"],
                     "st_route_continuity": rs["route_continuity"], "train_time_s": t_fit,
                     "inference_time_s": t_inf, "inference_us_per_candidate": 1e6 * t_inf / len(st_c)})
        if tag == "tuned":
            joblib.dump(m, ROOT / "models" / f"{model_name}_tuned.joblib", compress=3)
            out = st_c[["trip_id", "pt", "arc", "frac", "label", "reliable"]].copy()
            out["score"] = p
            out.to_pickle(RES / "preds" / f"student_{model_name}_tuned.pkl")
    cmp = pd.DataFrame(rows)
    cmp.to_csv(RES / f"bayes_opt_tuned_vs_untuned_{model_name}.csv", index=False)
    (RES / f"bayes_opt_best_{model_name}.json").write_text(json.dumps({
        "model": model_name, "objective": "mean 3-fold GroupKFold observation Top-1 accuracy (train trips)",
        "sampler": "Optuna TPE, seed 42, 8 random start-up trials", "n_trials": n_trials,
        "best_value": study.best_value, "best_params": study.best_params, "search_seconds": search_s,
    }, indent=2))
    print(cmp.to_string())

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    v = trials["value"].to_numpy()
    ax[0].plot(v, "o", alpha=0.6, label="trial"); ax[0].plot(np.maximum.accumulate(v), "r-", label="best so far")
    ax[0].set(title=f"Bayesian optimisation of {model_name}", xlabel="trial", ylabel="CV Top-1 accuracy"); ax[0].legend()
    xs = np.arange(4); w = 0.38
    metrics = ["cv_macro_f1", "st_top1_acc", "st_macro_f1", "st_route_continuity"]
    ax[1].bar(xs - w / 2, cmp.loc[0, metrics].astype(float), w, label="untuned")
    ax[1].bar(xs + w / 2, cmp.loc[1, metrics].astype(float), w, label="tuned")
    ax[1].set(xticks=xs, ylim=(0.8, 1.0), title="Untuned vs tuned"); ax[1].set_xticklabels(metrics, rotation=15); ax[1].legend()
    fig.tight_layout(); fig.savefig(RES / "figures" / "bayes_opt.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="defaults to best CV Top-1 model in results/model_comparison.csv")
    ap.add_argument("--trials", type=int, default=30)
    a = ap.parse_args()
    name = a.model
    if name is None:
        cmp = pd.read_csv(RES / "model_comparison.csv")
        name = cmp.sort_values("cv_top1_mean", ascending=False)["model"].iloc[0]
        print("selected by CV Top-1:", name)
    main(name, a.trials)
