"""Train and compare every model in MODEL_ZOO.

Protocol
  * 5-fold GroupKFold on the 80 training trips (groups = trip id, no trip spans folds).
  * Fit on all 80 training trips, evaluate on the 10 student-testing trips.
  * Training uses only candidates of fixes with a reliable reference label.
Outputs: results/model_comparison.csv, results/cv_folds.csv, results/experiments/*.json,
         results/preds/student_<model>.pkl, results/figures/cmp_*.png, models/<name>.joblib
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
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from pathsnap.experiment import fit_model, load_partition, obs_top1, predict, route_scores, score_block  # noqa: E402
from pathsnap.io import ROOT  # noqa: E402
from pathsnap.models import MODEL_ZOO, SEED  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"


def main(models):
    np.random.seed(SEED)
    for d in (RES / "experiments", RES / "preds", FIG, ROOT / "models"):
        d.mkdir(parents=True, exist_ok=True)
    tr_c, tr_p = load_partition("train")
    st_c, st_p = load_partition("student_test")
    trips = tr_c["trip_id"].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(tr_c, groups=trips))
    fold_rows, summary = [], []
    for name in models:
        family, factory, max_rows, desc = MODEL_ZOO[name]
        print(f"== {name}", flush=True)
        cv = []
        for k, (i_tr, i_va) in enumerate(folds):
            va_c = tr_c.iloc[i_va]
            va_trips = set(va_c["trip_id"])
            m, t_fit, n_fit = fit_model(factory, tr_c.iloc[i_tr], max_rows)
            p, _ = predict(m, va_c)
            s = score_block(va_c, tr_p[tr_p["trip_id"].isin(va_trips)], p)
            s_tr = score_block(tr_c.iloc[i_tr].sample(min(len(i_tr), 200_000), random_state=SEED),
                               tr_p, predict(m, tr_c.iloc[i_tr].sample(min(len(i_tr), 200_000), random_state=SEED))[0])
            row = {"model": name, "fold": k, "fit_s": t_fit, "n_fit_rows": n_fit,
                   **{f"val_{x}": s[x] for x in ("macro_f1", "f1", "log_loss", "top1_acc")},
                   "train_macro_f1": s_tr["macro_f1"]}
            cv.append(row)
            print(f"   fold {k}: val macroF1={s['macro_f1']:.4f} top1={s['top1_acc']:.4f} "
                  f"train macroF1={s_tr['macro_f1']:.4f} ({t_fit:.1f}s)", flush=True)
        fold_rows += cv
        cvdf = pd.DataFrame(cv)

        model, t_fit, n_fit = fit_model(factory, tr_c, max_rows)
        p, t_inf = predict(model, st_c)
        s = score_block(st_c, st_p, p)
        pred_pts = obs_top1(st_c, st_p, p)
        rs = route_scores(pred_pts)
        out = st_c[["trip_id", "pt", "arc", "frac", "label", "reliable"]].copy()
        out["score"] = p
        out.to_pickle(RES / "preds" / f"student_{name}.pkl")
        joblib.dump(model, ROOT / "models" / f"{name}.joblib", compress=3)
        rec = {
            "model": name, "family": family, "config": desc, "seed": SEED,
            "cv_macro_f1_mean": cvdf["val_macro_f1"].mean(), "cv_macro_f1_std": cvdf["val_macro_f1"].std(),
            "cv_top1_mean": cvdf["val_top1_acc"].mean(), "cv_log_loss_mean": cvdf["val_log_loss"].mean(),
            "cv_train_macro_f1_mean": cvdf["train_macro_f1"].mean(),
            "st_accuracy": s["accuracy"], "st_precision": s["precision"], "st_recall": s["recall"],
            "st_f1": s["f1"], "st_macro_f1": s["macro_f1"], "st_log_loss": s["log_loss"],
            "st_top1_acc": s["top1_acc"], "st_top1_acc_all_labelled": s["top1_acc_all_labelled"],
            "st_route_continuity": rs["route_continuity"], "st_route_f1": rs["route_f1"],
            "train_time_s": t_fit, "n_train_rows": n_fit,
            "inference_time_s": t_inf, "inference_us_per_candidate": 1e6 * t_inf / len(st_c),
            "st_confusion": s["confusion"],
        }
        (RES / "experiments" / f"{name}.json").write_text(json.dumps(rec, indent=2, default=float))
        summary.append(rec)
        print(f"   student-test: top1={s['top1_acc']:.4f} macroF1={s['macro_f1']:.4f} "
              f"logloss={s['log_loss']:.4f} continuity={rs['route_continuity']:.4f}", flush=True)

    # merge with any earlier runs so models can be (re)run individually
    cmp_path, fold_path = RES / "model_comparison.csv", RES / "cv_folds.csv"
    new = pd.DataFrame(summary).drop(columns=["st_confusion"])
    if cmp_path.exists():
        old = pd.read_csv(cmp_path)
        new = pd.concat([old[~old["model"].isin(new["model"])], new])
    new.to_csv(cmp_path, index=False)
    newf = pd.DataFrame(fold_rows)
    if fold_path.exists():
        oldf = pd.read_csv(fold_path)
        newf = pd.concat([oldf[~oldf["model"].isin(newf["model"])], newf])
    newf.to_csv(fold_path, index=False)
    plots(new, newf)


def plots(cmp, folds):
    cmp = cmp.sort_values("st_top1_acc")
    order = list(cmp["model"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    data = [folds.loc[folds["model"] == m, "val_macro_f1"] for m in order]
    ax.boxplot(data, tick_labels=order)
    ax.scatter(range(1, len(order) + 1), cmp["cv_train_macro_f1_mean"], marker="x", color="red", label="train (in-fold) macro F1")
    ax.set(title="5-fold group CV macro F1 (80 training trips)", ylabel="macro F1")
    ax.legend(); plt.xticks(rotation=30, ha="right"); fig.tight_layout()
    fig.savefig(FIG / "cmp_cv_macro_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    xs = np.arange(len(order)); w = 0.38
    ax.bar(xs - w / 2, cmp["st_top1_acc"], w, label="Top-1 accuracy (fixes)")
    ax.bar(xs + w / 2, cmp["st_macro_f1"], w, label="Macro F1 (candidates)")
    lo = min(cmp["st_top1_acc"].min(), cmp["st_macro_f1"].min())
    ax.set(ylim=(max(0, lo - 0.05), 1), xticks=xs, title="Student-testing partition (10 trips)")
    ax.set_xticklabels(order, rotation=30, ha="right"); ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "cmp_student_top1_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].barh(order, cmp["st_log_loss"]); ax[0].set(title="Student-test log loss (lower is better)", xscale="log")
    ax[1].barh(order, cmp["inference_us_per_candidate"]); ax[1].set(title="Inference time per candidate (µs)", xscale="log")
    fig.tight_layout(); fig.savefig(FIG / "cmp_logloss_inference.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODEL_ZOO))
    main(ap.parse_args().models)
