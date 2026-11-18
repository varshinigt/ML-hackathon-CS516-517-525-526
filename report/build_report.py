"""Build report/report.html (and report.pdf via report/to_pdf.js) from the saved results.

Every number and figure in the report is read from results/ or artifacts/, so the report can
be regenerated after any re-run. Narrative text that interprets the results lives in
report/narrative.py and should be reviewed and rewritten by the team in their own words.
"""
import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "report"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import narrative as N  # noqa: E402

RES, FIG, IMG = ROOT / "results", ROOT / "results" / "figures", ROOT / "report" / "img"


def pct(v, d=2):
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.{d}f}%"


def f3(v):
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def table(df, cols=None, fmt=None):
    fmt = fmt or {}
    cols = cols or list(df.columns)
    h = "<table><thead><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in cols) + "</tr></thead><tbody>"
    for _, r in df.iterrows():
        h += "<tr>" + "".join(f"<td>{fmt.get(c, lambda x: html.escape(str(x)))(r[c])}</td>" for c in cols) + "</tr>"
    return h + "</tbody></table>"


def img(path, caption, width="100%"):
    rel = Path(path).relative_to(ROOT)
    return f'<figure><img src="../{rel.as_posix()}" style="width:{width}"><figcaption>{caption}</figcaption></figure>'


def confusion_fig(cmp):
    names = list(cmp["model"])
    fig, axes = plt.subplots(2, (len(names) + 1) // 2, figsize=(3.0 * ((len(names) + 1) // 2), 6))
    for ax, n in zip(axes.ravel(), names):
        cm = np.array(json.loads((RES / "experiments" / f"{n}.json").read_text())["st_confusion"])
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}\n({cmn[i, j]:.2f})", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > .5 else "black")
        ax.set(title=n, xticks=[0, 1], yticks=[0, 1], xlabel="predicted", ylabel="true")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("Student-test candidate confusion matrices (threshold 0.5, reliable fixes)")
    fig.tight_layout(); fig.savefig(FIG / "cmp_confusion.png", dpi=130); plt.close(fig)


def mlp_curve():
    import joblib
    m = joblib.load(ROOT / "models" / "MLP.joblib")
    mlp = m[-1]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(mlp.loss_curve_, label="training loss (log loss)")
    ax2 = ax.twinx()
    ax2.plot(mlp.validation_scores_, color="tab:orange", label="validation accuracy (10% hold-out)")
    ax.set(xlabel="epoch", ylabel="training log loss", title="MLP training history (only model with epochs)")
    fig.legend(loc="center right"); fig.tight_layout(); fig.savefig(FIG / "mlp_history.png", dpi=130); plt.close(fig)
    return len(mlp.loss_curve_)


def main():
    inv = pd.read_csv(RES / "eda" / "inventory.csv", dtype={"trip_id": str})
    plog = pd.read_csv(RES / "preprocessing_log.csv", dtype={"trip_id": str})
    cmp = pd.read_csv(RES / "model_comparison.csv")
    folds = pd.read_csv(RES / "cv_folds.csv")
    best_name = cmp.sort_values("cv_top1_mean", ascending=False)["model"].iloc[0]
    bo = json.loads((RES / f"bayes_opt_best_{best_name}.json").read_text())
    tvu = pd.read_csv(RES / f"bayes_opt_tuned_vs_untuned_{best_name}.csv")
    trials = pd.read_csv(RES / f"bayes_opt_trials_{best_name}.csv")
    hmm = pd.read_csv(RES / "hmm_comparison.csv")
    hper = pd.read_csv(RES / "hmm_per_trip.csv", dtype={"trip_id": str})
    hcfg = json.loads((ROOT / "artifacts" / "hmm_config.json").read_text())
    hgrid = pd.read_csv(RES / "hmm_grid.csv")
    confusion_fig(cmp)
    n_epochs = mlp_curve()

    # nearest-arc rule as a reference point (not one of the compared models)
    pts_st = pd.read_pickle(ROOT / "data" / "processed" / "points_student_test.pkl").reset_index().rename(columns={"index": "pt"})
    c_st = pd.read_pickle(ROOT / "data" / "processed" / "cands_student_test.pkl")
    near = c_st[c_st["dist_rank"] == 0][["trip_id", "pt", "arc"]].merge(pts_st, on=["trip_id", "pt"])
    rel = (near["ref_arc"] >= 0) & (near["ref_dist"] <= 50)
    nearest_top1 = float((near["arc"] == near["ref_arc"])[rel].mean())
    pts_tr = pd.read_pickle(ROOT / "data" / "processed" / "points_train.pkl")
    unreliable_tr = float((pts_tr["ref_dist"] > 50).mean())
    unreliable_st = float((pts_st["ref_dist"] > 50).mean())
    c_tr = pd.read_pickle(ROOT / "data" / "processed" / "cands_train.pkl")
    pos_rate = float(c_tr["label"].mean())

    ctx = dict(inv=inv, plog=plog, cmp=cmp, folds=folds, best=best_name, bo=bo, tvu=tvu, trials=trials,
               hmm=hmm, hper=hper, hcfg=hcfg, hgrid=hgrid, nearest_top1=nearest_top1, n_epochs=n_epochs,
               unreliable_tr=unreliable_tr, unreliable_st=unreliable_st, pos_rate=pos_rate,
               n_cands_tr=len(c_tr), n_cands_st=len(c_st))

    S = []
    S.append(f"""<h1>PathSnap: Map-Matching and Route Reconstruction</h1>
<p class="sub">UE24CS352A Machine Learning Hackathon, Technical Report. Team CS516 · CS517 · CS525 · CS526</p>
<div class="box"><b>Summary.</b> {N.summary(ctx)}</div>""")

    # ------------------------------------------------------------------ 2 EDA
    pc = inv.groupby("partition").agg(trips=("trip_id", "count"), gps_points=("n_gps", "sum"),
                                       median_fixes=("n_gps", "median"), median_arcs=("n_arcs", "median"),
                                       max_gap_s=("max_gap_s", "max"), missing_values=("missing_values", "sum")).reset_index()
    pc.loc[len(pc)] = ["eval_test", 10, "withheld", "-", "-", "-", "-"]
    anomalies = pd.read_csv(RES / "eda" / "anomalies.csv", dtype={"trip_id": str})
    S.append(f"""<h2>2. Dataset Understanding and Exploratory Data Analysis</h2>
<p class="note">Partitions used: all EDA uses the 80 training and 10 student-testing trips. The 10 evaluator trips were not available to us and are never read by any analysis or training script.</p>
<h3>2.1 Raw data and how the files relate</h3>{N.data_structure(ctx)}
{table(pc)}
<h3>2.2 How we preprocessed the data</h3>{N.preprocessing(ctx)}
<h3>2.3 Patterns observed during exploration</h3>{N.patterns(ctx)}
{img(FIG / 'eda_distributions.png', 'Fig 2.1: Fixes per trip, network size and reference route length (train and student_test).')}
{img(FIG / 'eda_sampling_intervals.png', 'Fig 2.2: Sampling intervals. Most steps are 1 s; long gaps exist (log scale).')}
<div class="two">{img(FIG / 'eda_all_trajectories.png', 'Fig 2.3: All GPS trajectories (train: blue, student_test: orange).')}
{img(FIG / 'eda_connectivity.png', 'Fig 2.4: Road-network connectivity per trip.')}</div>
<h4>Missing values and anomalies</h4>{N.missing(ctx)}
{table(anomalies.head(15), ['trip_id', 'partition', 'n_gps', 'max_gap_s', 'n_gaps_gt_10s', 'max_jump_m', 'route_discontinuities'], {'max_jump_m': lambda v: f'{v:.0f}'})}
<p class="note">First 15 of {len(anomalies)} flagged trips (gap &gt; 30 s, route discontinuity or jump &gt; 200 m). Full list: results/eda/anomalies.csv.</p>
<h3>2.4 Features extracted and why</h3>{N.features(ctx)}
{img(IMG / 'eda_trip_00000041.png', 'Fig 2.5: Folium map, training trip 00000041: GPS fixes (blue), road network near the trip (grey), reference route (green). Interactive HTML: results/maps/eda_trip_00000041.html. Basemap tiles were not reachable from the machine that rendered the screenshot.')}
""")

    # ------------------------------------------------------------------ 3 models
    cmp_sorted = cmp.sort_values("st_top1_acc", ascending=False)
    S.append(f"""<h2>3. Model Experimentation and Comparison</h2>{N.experiment_setup(ctx)}
<h3>3.1 Overall model comparison</h3>
{table(cmp_sorted, ['model', 'family', 'config', 'cv_macro_f1_mean', 'st_top1_acc', 'st_route_continuity', 'st_macro_f1', 'st_log_loss', 'inference_us_per_candidate'],
       {'cv_macro_f1_mean': f3, 'st_top1_acc': pct, 'st_route_continuity': pct, 'st_macro_f1': f3, 'st_log_loss': f3,
        'inference_us_per_candidate': lambda v: f'{v:.2f} µs/cand'})}
<p class="note">Columns: CV macro F1 = mean over 5 GroupKFold folds of the 80 training trips. Top-1 = observation-level Top-1 accuracy on student-test fixes with a reliable label (classifier only, no HMM). Route continuity = share of consecutive fixes whose predicted arcs are the same, adjacent or joined by ≤100 m of road. Macro F1 and log loss are on candidate rows. Inference time is per candidate row. For reference, always picking the nearest arc gives Top-1 = {pct(nearest_top1)}.</p>
{table(cmp_sorted, ['model', 'cv_macro_f1_std', 'cv_top1_mean', 'cv_train_macro_f1_mean', 'st_accuracy', 'st_precision', 'st_recall', 'st_f1', 'st_route_continuity_strict', 'st_route_f1', 'train_time_s', 'n_train_rows'],
       {c: f3 for c in ['cv_macro_f1_std', 'cv_top1_mean', 'cv_train_macro_f1_mean', 'st_accuracy', 'st_precision', 'st_recall', 'st_f1', 'st_route_continuity_strict', 'st_route_f1']} | {'train_time_s': lambda v: f'{v:.1f} s'})}
<p class="note">Additional metrics. Precision, recall and F1 are for the positive class ("this candidate is the reference arc"). Strict continuity counts only identical or directly adjacent arcs.</p>
<h3>3.2 Per-model analysis</h3>""")
    for _, r in cmp_sorted.iterrows():
        S.append(N.model_section(r, ctx))
    S.append(f"""<h3>3.3 Model visualisations</h3>
{img(FIG / 'cmp_cv_macro_f1.png', 'Fig 3.1: Cross-validation macro F1 for every model (box = 5 group folds; red x = in-fold training macro F1).')}
{img(FIG / 'cmp_student_top1_f1.png', 'Fig 3.2: Student-test Top-1 accuracy and macro F1.')}
{img(FIG / 'cmp_logloss_inference.png', 'Fig 3.3: Student-test log loss and inference cost.')}
{img(FIG / 'cmp_confusion.png', 'Fig 3.4: Confusion matrices (optional diagnostic).')}
{img(FIG / 'mlp_history.png', 'Fig 3.5: Training history of the MLP, the only model that trains in epochs. No curves are shown for the other models because they have no epoch history.', '75%')}
<h3>3.4 Model selection</h3>{N.selection(ctx)}""")

    # ------------------------------------------------------------------ 4 Bayes
    S.append(f"""<h2>4. Bayesian Hyperparameter Optimisation ({best_name})</h2>{N.bayes(ctx)}
{table(tvu, list(tvu.columns), {c: f3 for c in tvu.columns if c not in ('variant',)})}
{img(FIG / 'bayes_opt.png', f'Fig 4.1: Optuna TPE trials (objective = 3-fold group CV Top-1) and tuned vs untuned {best_name}.')}""")

    # ------------------------------------------------------------------ 5 HMM
    hcols = ["trip_id", "n_fixes", "clf_top1", "hmm_top1", "clf_continuity", "hmm_continuity", "clf_breaks", "hmm_breaks", "clf_route_f1", "hmm_route_f1"]
    S.append(f"""<h2>5. HMM and Viterbi Route Reconstruction</h2>{N.hmm(ctx)}
{table(hmm, list(hmm.columns), {c: f3 for c in hmm.columns if c != 'method'} | {'breaks_per_trip': lambda v: f'{v:.1f}'})}
<p class="note">Student-testing partition, final {best_name} model. Top-1 accuracy is pooled over all reliable fixes, and the other columns are means over trips.</p>
{table(hper[hcols], hcols, {c: f3 for c in hcols if c not in ('trip_id', 'n_fixes', 'clf_breaks', 'hmm_breaks')})}
{img(FIG / 'hmm_per_trip.png', 'Fig 5.1: Classifier-only vs classifier+HMM per student-test trip.')}
{img(FIG / 'hmm_grid.png', 'Fig 5.2: Decoder grid search on out-of-fold training scores.', '60%')}
{N.hmm_discussion(ctx)}""")

    # ------------------------------------------------------------------ 6 Folium
    st_maps = sorted(IMG.glob("student_trip_*.png"))
    worst = hper.sort_values("hmm_top1").iloc[0]["trip_id"]
    bestt = hper.sort_values("hmm_top1").iloc[-1]["trip_id"]
    S.append(f"""<h2>6. Geographical Visualisation Using Folium</h2>{N.folium_text(ctx)}
{img(IMG / f'student_trip_{bestt}.png', f'Fig 6.1: Student-test trip {bestt}: GPS (blue dots), road network (grey), reference (thick green), classifier-only (orange, dashed), HMM+Viterbi (red). Layer control top right. HTML: results/maps/student_trip_{bestt}.html')}
{img(IMG / f'student_trip_{worst}.png', f'Fig 6.2: Student-test trip {worst}, the hardest trip for the final pipeline. HTML: results/maps/student_trip_{worst}.html')}
<p class="note">Interactive maps for all {len(st_maps)} student-test trips and the EDA trips are in results/maps/. They are generated by scripts/05_hmm.py and scripts/01_eda.py using src/pathsnap/viz.py.</p>""")

    # ------------------------------------------------------------------ 7 API/UI
    log_lines = (ROOT / "report" / "api_console_sample.txt").read_text() if (ROOT / "report" / "api_console_sample.txt").exists() else ""
    S.append(f"""<h2>7. API, UI and Deployment</h2>{N.deployment(ctx)}
{img(IMG / 'ui.png', 'Fig 7.1: Evaluator UI (Laptop 2) after selecting a trip. During this rehearsal the API served the student-test trips.')}
<h4>API console output (Laptop 1)</h4><pre class="term">{html.escape(log_lines)}</pre>
<p class="note">Rehearsal log from <code>uvicorn api.main:app</code>. <b>Replace with the screenshot of the receiving laptop's terminal taken during the evaluation.</b></p>""")

    S.append(f"<h2>8. Conclusions</h2>{N.conclusions(ctx)}<h2>9. AI-assistance disclosure</h2>{N.ai_disclosure(ctx)}")

    css = (ROOT / "report" / "style.css").read_text()
    out = f"<!doctype html><html><head><meta charset='utf-8'><title>PathSnap Technical Report</title><style>{css}</style></head><body>{''.join(S)}</body></html>"
    (ROOT / "report" / "report.html").write_text(out)
    print("wrote report/report.html")
    try:
        subprocess.run(["node", str(ROOT / "report" / "to_pdf.js")], check=True)
    except Exception as e:  # PDF needs node + playwright
        print("PDF step skipped:", e)


if __name__ == "__main__":
    main()
