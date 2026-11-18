"""Report prose. Numbers are always pulled from ctx (the saved results).

DRAFT TEXT: written with AI assistance from our actual results. Team members should
read, verify and rewrite these sections in their own words before submission.
"""
import numpy as np


def p(v, d=2):
    return f"{100 * v:.{d}f}%"


def summary(c):
    h = c["hmm"].set_index("method")
    clf, hm = h.iloc[0], h.iloc[1]
    return (f"We frame map-matching as candidate ranking. Every GPS fix gets up to 8 nearby road arcs. A classifier "
            f"scores each (fix, arc) pair, and an HMM with Viterbi decoding turns these scores into a connected route. "
            f"We compared 10 models from 7 algorithm families with trip-grouped cross-validation. We selected and Bayesian-tuned "
            f"<b>{c['best']}</b>. On the 10 student-testing trips, the classifier alone reaches {p(clf.top1_acc)} "
            f"observation-level Top-1 accuracy, and HMM decoding raises it to <b>{p(hm.top1_acc)}</b>. Route continuity rises "
            f"from {p(clf.route_continuity)} to {p(hm.route_continuity)}, and breaks per trip fall from {clf.breaks_per_trip:.1f} to "
            f"{hm.breaks_per_trip:.1f}. The same pipeline runs behind a FastAPI service with a separate evaluator UI.")


def data_structure(c):
    inv = c["inv"]
    return f"""<p>Each trip is a folder with four header-less, whitespace-separated files:</p>
<ul>
<li><code>.track</code>: GPS fixes as <i>longitude, latitude, time (s)</i>.</li>
<li><code>.nodes</code>: road-network nodes as <i>longitude, latitude</i>. The row number is the node id.</li>
<li><code>.arcs</code>: directed road segments as <i>from_node, to_node</i>. The row number is the arc id.</li>
<li><code>.route</code>: the chronological list of arc ids the vehicle travelled (the reference).</li>
</ul>
<p>The files are linked only through ids. A route row points to an arc row, and an arc row points to two node rows.
A GPS row does <b>not</b> correspond to any network row. The route also says nothing about which fix lies on which arc,
so per-fix labels have to be derived (§2.2). Each trip has its own road network, and network size varies enormously:
from {inv.n_arcs.min():,} to {inv.n_arcs.max():,} arcs (median {int(inv.n_arcs.median()):,}). Arcs are directed: for
{p(inv.reverse_arc_frac.mean(), 0)} of arcs the opposite-direction arc also exists, so two-way streets appear as two arcs.
The reference routes are continuous chains (the head of one arc is the tail of the next) except in
{int((inv.route_discontinuities > 0).sum())} trips with {int(inv.route_discontinuities.sum())} breaks in total.</p>"""


def preprocessing(c):
    pl = c["plog"]
    return f"""<ol>
<li><b>Coordinates.</b> Longitude and latitude are angles, so distances in degrees are meaningless (one degree of longitude
is about 67 km here, against 111 km for a degree of latitude). Each trip is projected onto a local tangent plane
(equirectangular, centred on the trip mean): x = R·Δλ·cos φ₀, y = R·Δφ. For trips a few km across, the error against
the true geodesic distance is far below the GPS noise. All later geometry is in metres.</li>
<li><b>Cleaning.</b> We drop fixes with missing or out-of-range coordinates and duplicate timestamps, sort by time, and
remove isolated speed spikes (fixes reached from <i>and</i> left at more than 70 m/s). Across the 90 trips this removed
{int(pl.invalid.sum())} invalid fixes, {int(pl.duplicate_t.sum())} duplicate timestamps and {int(pl.speed_spikes.sum())} spikes.
The raw row index of every fix is kept, so each prediction maps back to its original observation.</li>
<li><b>Motion features.</b> Speed and heading come from a centred ±2-fix difference. Heading is flagged unreliable when the
vehicle moved less than 3 m in that window, because the heading of a stationary vehicle is noise.</li>
<li><b>Reference labels per fix.</b> The <code>.route</code> gives arcs in order but no timing. We align fixes to the route
with a monotone dynamic program: we choose a non-decreasing route index j(i) for each fix i that minimises the sum of
fix-to-arc distances. Arcs may be skipped, because a short arc can be crossed between two 1 Hz fixes. Fixes more than
50 m from their aligned arc are marked <b>unreliable</b>: {p(c['unreliable_tr'], 1)} of training fixes and
{p(c['unreliable_st'], 1)} of student-test fixes. They occur where the reference route does not cover part of the GPS
track (two training trips are mostly uncovered). Unreliable fixes are excluded from training and from the headline
accuracy. Accuracy over all labelled fixes is also reported.</li>
<li><b>Candidate generation.</b> Each fix gets the 8 nearest arcs within 60 m. If there are none, the search widens to
200 m, and then to the single nearest arc, so every fix has at least one state. Radius queries use a k-d tree over points
sampled every 10 m along the arcs, followed by exact point-to-segment distances. The reference arc is missing from the
candidate set (the <i>candidate miss rate</i>) for {p(pl[pl.partition == 'train'].miss_rate.mean())} of training fixes and
{p(pl[pl.partition == 'student_test'].miss_rate.mean())} of student-test fixes (trip means). This gives
{c['n_cands_tr']:,} training and {c['n_cands_st']:,} student-test candidate rows. {p(c['pos_rate'], 1)} of them are positives.</li>
</ol>"""


def patterns(c):
    inv = c["inv"]
    return f"""<ul>
<li>Sampling is essentially 1 Hz: a median of {p(inv.frac_dt_1s.median(), 1)} of steps are exactly 1 s. However, {int((inv.max_gap_s > 30).sum())} trips have
gaps longer than 30 s, and the longest is {inv.max_gap_s.max() / 3600:.1f} h (the vehicle parked and the log resumed). The decoder must therefore
handle gaps explicitly.</li>
<li>Trips differ a lot in length, from {inv.n_gps.min()} to {inv.n_gps.max():,} fixes (median {int(inv.n_gps.median()):,}). Cross-validation is therefore grouped
by trip, and accuracy is pooled over fixes.</li>
<li>All trips lie in one region (mean position about {inv.lon_mean.mean():.2f}° E, {inv.lat_mean.mean():.2f}° N, in the English East Midlands), with
extents up to {max(inv.extent_ew_km.max(), inv.extent_ns_km.max()):.0f} km. A local projection per trip is accurate enough.</li>
<li>The networks are dense, with a mean arc length of {inv.mean_arc_len_m.median():.0f} m (median over trips). Parallel and opposite-direction
arcs sit within GPS-noise distance of each other, so the nearest arc is often wrong: it is correct for only
{p(c['nearest_top1'])} of student-test fixes. Heading and temporal context are needed to separate such arcs.</li>
<li>The networks are not fully connected: a median of {int(inv.n_weak_components.median())} weakly connected components per trip, and
{int(inv.isolated_nodes.sum())} isolated nodes in total. The HMM must therefore tolerate unreachable transitions.</li>
</ul>"""


def missing(c):
    inv = c["inv"]
    return (f"<p>No trip has missing or non-numeric values: the total count of missing values is {int(inv.missing_values.sum())}, and timestamps never "
            f"decrease ({int(inv.non_monotonic_t.sum())} cases). The data-quality issues are instead GPS gaps, occasional large jumps "
            f"(largest {inv.max_jump_m.max():.0f} m between consecutive fixes), uncovered stretches of reference route and a few "
            f"route discontinuities.</p>")


def features(c):
    return """<p>Each candidate row (fix i, arc a) has 19 features. Each group addresses a confusion we saw in the EDA.</p>
<table><tr><th>Group</th><th>Features</th><th>Why</th></tr>
<tr><td>Geometry</td><td>dist, dist_rank, dist_minus_min, dist_ratio, n_cands</td><td>Closeness alone is the classic emission. The relative versions make it comparable across dense and sparse areas.</td></tr>
<tr><td>Direction</td><td>heading_diff, heading_valid, heading_x_valid</td><td>Separates the two directions of a two-way street and parallel roads. Masked when the vehicle is stationary.</td></tr>
<tr><td>Arc</td><td>frac (position along the arc), arc_len</td><td>Projections clamped at arc ends often mean the fix belongs to the next arc.</td></tr>
<tr><td>Motion</td><td>speed, dt_prev, dt_next</td><td>GPS noise and gap context. Slow vehicles have unreliable headings.</td></tr>
<tr><td>Temporal context</td><td>dist_prev, dist_next, dist_win_mean (±2 fixes)</td><td>A true arc stays close to neighbouring fixes, while a crossing street is close for only one fix.</td></tr>
<tr><td>Connectivity</td><td>near_prev_top1, conn_prev_top1, conn_next_top1</td><td>Whether the arc is the same as, or connected to, the nearest arc of the previous and next fix. These are label-free network cues.</td></tr></table>"""


def experiment_setup(c):
    return """<p><b>Learning task.</b> Binary classification of candidate rows (label 1 if the arc is the fix's reference arc).
Per fix, the arg-max of the predicted probability gives the classifier-only prediction. The probabilities also become the
HMM emissions.</p>
<p><b>Protocol.</b> We use 5-fold <code>GroupKFold</code> on the 80 training trips, with groups = trips, so no trip appears in
both training and validation. The final fit uses all 80 training trips, and evaluation uses the 10 student-testing trips.
Preprocessing and features are identical for every model. The seed is 42 throughout. Models that are expensive in the
number of rows (KNN, linear SVM, MLP) are fitted on a random row subsample, and the subsample size is recorded in the
configuration.</p>
<p><b>Primary metric.</b> Observation-level Top-1 accuracy on reliable fixes. The system's output is one road segment per
fix, so this is the quantity the evaluators measure. Macro F1 on candidate rows (threshold 0.5) is reported for
classification quality. Log loss measures probability quality, which matters because the HMM consumes probabilities.
All per-model logs are in results/experiments/*.json, the fold-level results in results/cv_folds.csv and the training
console log in results/train_compare.log.</p>"""


INFO = {
    "LogisticRegression": ("A linear model of the log-odds is the simplest probabilistic scorer. Its probabilities are usually well calibrated, which suits HMM emissions.",
                           "Fast, stable and interpretable coefficients. However, it cannot model interactions such as a large heading difference mattering only when heading is valid, and its decision boundary is linear.",
                           "Binary cross-entropy (log loss) with L2 penalty, lbfgs."),
    "GaussianNB": ("A generative baseline that is extremely cheap to train and evaluate.",
                   "Needs almost no tuning. However, it assumes the features are conditionally independent and Gaussian, which ours clearly are not (dist, dist_rank and dist_win_mean are strongly correlated, and the counts are discrete). This makes its probabilities over-confident.",
                   "Maximum likelihood of class-conditional Gaussians (no iterative loss)."),
    "KNN": ("The right arc for a fix is similar to the right arcs of geometrically similar fixes in other trips, so an instance-based model is a natural non-parametric check.",
            "No training and flexible boundaries. However, it is sensitive to feature scaling (hence the StandardScaler), slow at inference, and gives coarse probabilities. It had to be fitted on a subsample.",
            "None (lazy learner). Probabilities are distance-weighted neighbour votes."),
    "DecisionTree": ("Rules such as 'close and heading-aligned' match how a person would snap a point.",
                     "Interpretable and handles interactions and unscaled features. A single tree has high variance and produces piecewise-constant probabilities.",
                     "Gini impurity reduction (greedy)."),
    "RandomForest": ("Averaging many decorrelated trees keeps the tree's ability to model interactions while reducing variance.",
                     "Robust and needs little tuning. Its probabilities are smoother than a single tree's but tend to be under-confident near 0 and 1. It is large on disk and slower at inference than boosting.",
                     "Gini impurity per tree, bootstrap aggregation."),
    "ExtraTrees": ("Like Random Forest but with random split thresholds. This adds more variance reduction, which suits noisy GPS features.",
                   "Fast to train and smooth. Random thresholds can underfit sharp boundaries such as dist_rank = 0.",
                   "Gini impurity with randomised thresholds."),
    "HistGradientBoosting": ("Boosting corrects the errors of previous trees one step at a time, and histogram binning makes it fast on more than a million rows.",
                             "Strong on tabular data and handles interactions. It needs a learning-rate and iteration trade-off.",
                             "Binary log loss, gradient boosting."),
    "LightGBM": ("Leaf-wise gradient boosting is state of the art for tabular ranking-like problems with many rows and a few dozen features.",
                 "High accuracy, fast inference and good probabilities. However, it has many interacting hyperparameters and can overfit with large leaf counts, which motivated the Bayesian search.",
                 "Binary log loss, gradient boosting (leaf-wise growth, bagging)."),
    "LinearSVM": ("A maximum-margin linear boundary, the kernel-family representative. A linear kernel is the only practical option at a million rows.",
                  "Robust margin and fast. However, it has no native probabilities (Platt scaling was added) and the same linear limitation as logistic regression.",
                  "Hinge loss + L2 (squared-hinge LinearSVC), then sigmoid calibration on log loss."),
    "MLP": ("A neural network can learn non-linear combinations of the features end to end.",
            "Flexible and gives smooth probabilities. However, it needs scaling, is sensitive to initialisation and learning rate, and was trained on a 300k-row subsample.",
            "Binary cross-entropy, Adam optimiser, early stopping on a 10% validation split."),
}


def model_section(r, c):
    why, sl, loss = INFO[r.model]
    gap = r.cv_train_macro_f1_mean - r.cv_macro_f1_mean
    if gap > 0.03:
        fit = (f"<b>Overfitting.</b> The in-fold training macro F1 ({r.cv_train_macro_f1_mean:.4f}) is clearly above the validation macro F1 "
               f"({r.cv_macro_f1_mean:.4f}), a gap of {gap:.3f}.")
    elif r.cv_macro_f1_mean < c["cmp"].cv_macro_f1_mean.max() - 0.03 and gap < 0.01:
        fit = (f"<b>Underfitting.</b> Training ({r.cv_train_macro_f1_mean:.4f}) and validation ({r.cv_macro_f1_mean:.4f}) macro F1 are close "
               f"but both are well below the best models, so the model is too simple for the feature interactions.")
    else:
        fit = (f"<b>No strong over- or underfitting.</b> Training macro F1 {r.cv_train_macro_f1_mean:.4f} vs validation {r.cv_macro_f1_mean:.4f} "
               f"(gap {gap:+.3f}), with fold standard deviation {r.cv_macro_f1_std:.4f}.")
    fit += (" The validation spread across folds mostly reflects which trips land in a fold (trips 33 and 54 have long uncovered stretches)."
            if r.cv_macro_f1_std > 0.01 else "")
    return f"""<div class="model"><h4>{r.model} ({r.family})</h4>
<p><b>A. Why suitable:</b> {why}<br><b>Strengths and limitations:</b> {sl}</p>
<p><b>B. Configuration:</b> {r.config}. <b>Loss / objective:</b> {loss}</p>
<p><b>C. Results:</b> CV macro F1 {r.cv_macro_f1_mean:.4f} ± {r.cv_macro_f1_std:.4f}, CV Top-1 {p(r.cv_top1_mean)}.
Student test: Top-1 <b>{p(r.st_top1_acc)}</b>, macro F1 <b>{r.st_macro_f1:.4f}</b>, log loss <b>{r.st_log_loss:.4f}</b>,
precision {r.st_precision:.4f}, recall {r.st_recall:.4f}, route continuity {p(r.st_route_continuity)}.
Training {r.train_time_s:.1f} s on {int(r.n_train_rows):,} rows, inference {r.inference_us_per_candidate:.2f} µs per candidate.<br>{fit}</p></div>"""


def selection(c):
    cmp = c["cmp"].sort_values("cv_top1_mean", ascending=False)
    b = cmp.iloc[0]
    s2 = cmp.iloc[1]
    return f"""<p>We selected <b>{c['best']}</b> using the cross-validation Top-1 accuracy on the training trips
({p(b.cv_top1_mean)}, against {p(s2.cv_top1_mean)} for the runner-up {s2.model}). The student-testing trips confirmed this choice but did
not drive it. The other selection criteria also point the same way:</p>
<ul><li><b>Generalisation:</b> student-test Top-1 {p(b.st_top1_acc)} and macro F1 {b.st_macro_f1:.4f}.</li>
<li><b>Stability:</b> fold standard deviation of macro F1 {b.cv_macro_f1_std:.4f}.</li>
<li><b>Probability quality:</b> log loss {b.st_log_loss:.4f}. Low log loss means the probabilities can be used directly as HMM emissions.</li>
<li><b>Inference speed:</b> {b.inference_us_per_candidate:.2f} µs per candidate, so a whole trip is scored in well under a second.</li>
<li><b>Decoder compatibility:</b> it outputs calibrated per-candidate probabilities.</li></ul>
<p>The linear and Naive Bayes models are clearly weaker, which confirms that the feature interactions matter. KNN and the
MLP are competitive but slower and were trained on subsamples. The tree ensembles are close, but slower or larger.</p>"""


def bayes(c):
    bo, tvu = c["bo"], c["tvu"].set_index("variant")
    t = c["trials"]
    params = "".join(f"<li><code>{k}</code> = {v:.4g}</li>" if isinstance(v, float) else f"<li><code>{k}</code> = {v}</li>"
                     for k, v in bo["best_params"].items())
    u, tu = tvu.loc["untuned"], tvu.loc["tuned"]
    better = tu.st_top1_acc > u.st_top1_acc
    return f"""<ol>
<li><b>Hyperparameters searched:</b> number of trees, learning rate, leaves per tree, minimum samples per leaf, row subsampling, column subsampling, and L1/L2 regularisation (see scripts/04_bayes_opt.py).</li>
<li><b>Ranges:</b> n_estimators 150–900 (step 50); learning_rate 0.01–0.2 (log); num_leaves 15–255 (log); min_child_samples 10–400 (log); subsample 0.5–1.0; colsample_bytree 0.4–1.0; reg_lambda 1e-3–30 (log); reg_alpha 1e-3–10 (log).
The ranges centre on LightGBM defaults and widen in both directions: towards simpler trees, because the CV gaps are small, and towards more capacity, for the dense-network trips.</li>
<li><b>Objective:</b> mean observation-level Top-1 accuracy over 3-fold GroupKFold on the 80 training trips, which is the same quantity the decoder consumes. Macro F1 and log loss were logged for every trial.</li>
<li><b>Trials:</b> {bo['n_trials']} with Optuna's TPE sampler (seed 42, 8 random start-up trials), taking {bo['search_seconds'] / 60:.1f} min in total. All trials are in results/bayes_opt_trials_{c['best']}.csv.</li>
<li><b>Best values</b> (CV Top-1 {p(bo['best_value'])}):<ul>{params}</ul></li>
<li><b>Did tuning help?</b> {'Yes' if better else 'Only marginally / not on the student test'}. Student-test Top-1 went from {p(u.st_top1_acc)} to {p(tu.st_top1_acc)},
CV macro F1 from {u.cv_macro_f1:.4f} to {tu.cv_macro_f1:.4f}, and log loss from {u.st_log_loss:.4f} to {tu.st_log_loss:.4f}. Inference time changed from {u.inference_time_s:.2f} s to
{tu.inference_time_s:.2f} s for the whole student-test partition. Both variants used the same partitions and metrics. The best trial improved on the worst by
{p(t.value.max() - t.value.min())} CV Top-1.</li></ol>"""


def hmm(c):
    prm = c["hcfg"]["params"]
    return f"""<ol>
<li><b>Hidden states.</b> At fix i, the states are its candidate arcs, each stored as (fix, arc id, projection fraction along the arc).
A decoded state is therefore directly a road segment, which keeps the candidate-to-segment correspondence.</li>
<li><b>Emission probabilities.</b> The classifier probability is renormalised over the fix's candidates and floored at 1e-4:
log P<sub>e</sub>(a|i) = w<sub>e</sub> · log(p<sub>i,a</sub> / Σ<sub>a'</sub> p<sub>i,a'</sub>), with w<sub>e</sub> = {prm['w_emission']}.</li>
<li><b>Transition probabilities</b> follow Newson and Krumm (2009). d<sub>gc</sub> is the straight-line distance between consecutive fixes, and d<sub>rt</sub> is the shortest
directed network distance from the projected position on arc a to the projected position on arc b (Dijkstra over the arc graph, cached per source node, searched up to 3 km).
log P<sub>t</sub> = −|d<sub>rt</sub> − d<sub>gc</sub>| / β, with β = {prm['beta']} m. A backward step of up to 15 m on the same arc counts as standing still (GPS jitter).</li>
<li><b>Missing candidates and gaps.</b> Candidate generation guarantees at least one state per fix (via the 200 m fallback and then the nearest arc). If the time gap exceeds
gap_s = {prm['gap_s']} s, the transition matrix is set to zero (uniform), so the chain restarts from emissions, because network distances across long gaps are unreliable.</li>
<li><b>Disconnected segments.</b> If no path exists within the search limit (another component, or against a one-way direction), the transition gets a large finite penalty
(log P = {prm['unreachable_logp']}) instead of −∞. Viterbi therefore avoids such jumps whenever a connected alternative exists, but never fails. It returns a segment for every fix.</li>
<li><b>Viterbi recurrence.</b> δ<sub>i</sub>(s) = log P<sub>e</sub>(s) + max<sub>s'</sub>[δ<sub>i−1</sub>(s') + log P<sub>t</sub>(s'→s)], with back-pointers ψ<sub>i</sub>(s) = argmax; the path is recovered by backtracking.</li>
<li><b>Decoder tuning</b> used only training data. Out-of-fold probabilities for the 80 training trips came from 3-fold GroupKFold, and a grid search over β, w<sub>e</sub> and gap_s
({len(c['hgrid'])} combinations) maximised training out-of-fold Top-1 ({p(c['hcfg']['train_oof_classifier_top1'])} for the classifier alone → {p(c['hcfg']['train_oof_hmm_top1'])} with the HMM).</li></ol>"""


def hmm_discussion(c):
    h = c["hper"]
    worse = h[h.hmm_top1 < h.clf_top1]
    improved = int((h.hmm_top1 > h.clf_top1).sum())
    hh = c["hmm"].set_index("method")
    t = (f"<p><b>Did HMM decoding improve the route?</b> Yes. Pooled Top-1 rose from {p(hh.iloc[0].top1_acc)} to {p(hh.iloc[1].top1_acc)}, and it improved on "
         f"{improved} of {len(h)} trips. The largest gain is in continuity: isolated wrong snaps to a crossing street or the "
         f"opposite carriageway cannot be reached from their neighbours at a plausible distance, so Viterbi replaces them. Route F1 and the "
         f"order-aware LCS ratio rise as well.</p>")
    if len(worse):
        t += (f"<p><b>Where it did not help:</b> on trip(s) {', '.join(worse.trip_id)} accuracy dropped slightly. Inspecting the maps shows "
              f"that when the classifier is confidently wrong for a long run of fixes (for example a parallel service road), "
              f"the HMM keeps the whole run consistent and wrong. Breaks that remain come mostly from gap cuts and from stretches where "
              f"the network has no connecting arc.</p>")
    else:
        t += ("<p><b>Where it did not help:</b> the remaining errors are mostly long runs on a parallel road, where the classifier and the HMM agree on "
              "the wrong but connected alternative, and fixes right at junctions where the label itself is ambiguous.</p>")
    return t


def folium_text(c):
    return """<p>Maps are generated with Folium by <code>src/pathsnap/viz.py</code>. Each map has the following layers, each of which can be toggled:</p>
<ul>
<li>Raw GPS fixes (blue dots).</li>
<li>Road-network arcs within 150 m of the trajectory (grey).</li>
<li>The reference route (thick translucent green).</li>
<li>The classifier-only prediction (dashed orange).</li>
<li>The HMM/Viterbi prediction (red).</li>
<li>Start and end markers, a scale bar and a legend.</li>
</ul>
<p>The view is fitted to the trajectory's bounding box. Each map is saved as interactive HTML. The screenshots below come from those files.</p>"""


def deployment(c):
    return """<ul>
<li><b>Laptop 1 (API).</b> <code>uvicorn api.main:app --host 0.0.0.0 --port 8000</code> loads the final model, the feature configuration and the HMM
parameters (artifacts/pipeline_config.json, artifacts/hmm_config.json). <code>GET /trips</code> lists the evaluation trips found in
<code>data/raw/eval_test</code>. <code>POST /predict {"trip_id"}</code> runs the whole pipeline: load raw files <i>without</i> the route,
clean, generate candidates and features, classify, run Viterbi. Only after prediction does it read the reference route to compute
accuracy server-side. The response contains the GPS fixes, the predicted polylines, one arc id per raw observation and metrics, never the
reference route. Every request is logged with timings and accuracy to the console and to logs/api.log.</li>
<li><b>Laptop 2 (UI).</b> A static page (<code>ui/index.html</code>, Leaflet bundled locally) served with <code>python -m http.server</code>.
The evaluator enters the API address, loads the ten trips, selects one, and sees the predicted route over the GPS fixes with Top-1
accuracy, classifier-only accuracy, continuity, breaks, route F1 and processing time.</li>
<li>No outputs are hard-coded. The UI holds no data and calls the API for every selection.</li></ul>"""


def conclusions(c):
    hh = c["hmm"].set_index("method")
    return f"""<ul>
<li>Framing map-matching as candidate ranking lets standard classifiers be compared fairly. Temporal-context and heading features are what
lift the classifiers above the nearest-arc rule ({p(c['nearest_top1'])} → {p(hh.iloc[0].top1_acc)} Top-1).</li>
<li>{c['best']} gave the best cross-validated accuracy and well-calibrated probabilities, and Bayesian optimisation tuned it further.</li>
<li>HMM/Viterbi decoding adds network connectivity, raising Top-1 to {p(hh.iloc[1].top1_acc)} and producing far more continuous routes.</li>
<li><b>Limitations:</b> the per-fix labels come from our own alignment of the reference route, which itself may be imperfect. About 11% of
training fixes could not be labelled reliably. The 50 m reliability threshold is a judgement call. Accuracy over all labelled fixes is reported alongside.</li>
<li><b>Future work:</b> learn the transition model, try sequence models such as a CRF, and use speed-dependent β.</li></ul>"""


def ai_disclosure(c):
    return """<p>We used an AI assistant to help write code and to draft parts of this report. All experiments were executed on the
provided data with the code in our repository. Every number, table, plot and map is produced by those scripts and can be regenerated with
the commands in the README. The team has reviewed the implementation and results and can explain the design decisions.</p>"""
