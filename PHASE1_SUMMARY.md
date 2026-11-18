# PathSnap: Phase 1 summary

These are results from the first phase: data exploration, preprocessing and features, model experimentation, Bayesian optimisation, and a working API and UI.
Every number below is read from the files in `results/`. Reproduce them with the scripts listed in the README.

## 1. Dataset (train and student_test only; the evaluator trips are withheld)

| partition    |   trips |   gps_fixes |   median_fixes |   median_arcs |   max_gap_s |
|:-------------|--------:|------------:|---------------:|--------------:|------------:|
| student_test |      10 |       19160 |           1420 |       13145   |         210 |
| train        |      80 |      208417 |           1529 |       19526.5 |      153064 |

- Each trip has four files: `.track` (lon, lat, t), `.nodes` (lon, lat; row = node id), `.arcs` (from, to; row = arc id) and `.route` (the ordered reference arc ids).
- No values are missing and timestamps never go backwards. About 97% of steps are 1 s, but 30 trips have gaps longer than 30 s, and the longest is 42.5 h.
- Network size varies from 649 to 307,392 arcs. Arcs are directed, and 92% of them have a reverse twin.
- Plots: `results/figures/eda_*.png`. Folium maps: `results/maps/eda_trip_*.html`. Tables: `results/eda/`.

## 2. Preprocessing and labels

- **Projection.** Each trip is projected to a local tangent plane, so all distances are in metres. Euclidean distance in degrees is not valid.
- **Cleaning.** Invalid fixes, duplicate timestamps and speed spikes faster than 70 m/s are removed. The raw index of every fix is kept.
- **Labels.** The route has no timing, so each fix gets its reference arc from a monotone dynamic-programming alignment of the route to the GPS track. Fixes more than 50 m from their aligned arc are marked unreliable (11% of training fixes, mostly in two trips where the route covers only part of the track).
- **Candidates.** Each fix gets up to 8 arcs within 60 m, with fallbacks to 200 m and then to the nearest arc. The mean candidate miss rate is 2.78% for training and 0.42% for student-test trips.
- **Features.** 19 features per (fix, candidate): distance and rank, heading difference, position along the arc, speed and time gaps, the distances of neighbouring fixes, and connectivity to the neighbours' nearest arcs.

## 3. Model comparison (10 models, 7 families)

5-fold GroupKFold over the 80 training trips (folds split by trip), then a fit on all 80 and evaluation on the 10 student-test trips.
Top-1 means observation-level Top-1 accuracy (classifier only, before HMM). Macro F1 and log loss are computed on candidate rows.

| model                | family                 |   cv_macro_f1_mean |   cv_macro_f1_std |   cv_top1_mean |   st_top1_acc |   st_macro_f1 |   st_log_loss |   st_route_continuity |   train_time_s |   inference_us_per_candidate |
|:---------------------|:-----------------------|-------------------:|------------------:|---------------:|--------------:|--------------:|--------------:|----------------------:|---------------:|-----------------------------:|
| LightGBM             | Boosting               |             0.9265 |            0.0157 |         0.8804 |        0.8935 |        0.9339 |        0.0861 |                0.9845 |        11.0793 |                       5.0163 |
| HistGradientBoosting | Boosting               |             0.9252 |            0.0151 |         0.8795 |        0.8843 |        0.9289 |        0.0948 |                0.9843 |         6.198  |                       1.0912 |
| RandomForest         | Ensemble (bagging)     |             0.9252 |            0.0155 |         0.8792 |        0.8884 |        0.9308 |        0.0887 |                0.9835 |       159.272  |                       4.2452 |
| MLP                  | Neural network         |             0.9197 |            0.0154 |         0.8751 |        0.8787 |        0.924  |        0.0992 |                0.986  |        11.3594 |                       0.1861 |
| ExtraTrees           | Ensemble (bagging)     |             0.9183 |            0.0169 |         0.8742 |        0.876  |        0.9215 |        0.1095 |                0.9823 |        30.0031 |                       4.2367 |
| DecisionTree         | Tree-based             |             0.9209 |            0.016  |         0.8735 |        0.8889 |        0.928  |        0.098  |                0.981  |        16.9618 |                       0.0945 |
| LinearSVM            | Kernel-based           |             0.9009 |            0.0203 |         0.8681 |        0.867  |        0.9055 |        0.141  |                0.9764 |         1.7587 |                       0.2835 |
| LogisticRegression   | Linear / probabilistic |             0.9019 |            0.02   |         0.8676 |        0.8673 |        0.9064 |        0.1398 |                0.9777 |         2.484  |                       0.1432 |
| KNN                  | Instance-based         |             0.9095 |            0.0169 |         0.8642 |        0.868  |        0.9159 |        0.1432 |                0.9814 |         0.0135 |                      46.4729 |
| GaussianNB           | Linear / probabilistic |             0.8147 |            0.0195 |         0.8295 |        0.8246 |        0.8017 |        0.8879 |                0.9771 |         0.2928 |                       0.1506 |

**Selected model:** LightGBM, the best cross-validated Top-1. It also has the best student-test Top-1, macro F1 and log loss.
Figures: `results/figures/cmp_*.png`. Per-model configs and logs: `results/experiments/`.

## 4. Bayesian optimisation (Optuna TPE, 30 trials, objective = 3-fold group CV Top-1)

Best parameters: `{'n_estimators': 650, 'learning_rate': 0.011701505534591438, 'num_leaves': 214, 'min_child_samples': 257, 'subsample': 0.8987256635776459, 'colsample_bytree': 0.8810675677265736, 'reg_lambda': 5.294606726588959, 'reg_alpha': 0.16944963941901148}`

| variant   |   cv_macro_f1 |   cv_top1 |   st_top1_acc |   st_macro_f1 |   st_log_loss |   st_route_continuity |   train_time_s |   inference_time_s |   inference_us_per_candidate |
|:----------|--------------:|----------:|--------------:|--------------:|--------------:|----------------------:|---------------:|-------------------:|-----------------------------:|
| untuned   |        0.9281 |    0.8844 |        0.8935 |        0.9339 |        0.0861 |                0.9845 |        11.3359 |             0.614  |                       5.131  |
| tuned     |        0.9288 |    0.886  |        0.8974 |        0.9368 |        0.0845 |                0.9859 |        29.1203 |             3.4119 |                      28.5124 |

## 5. HMM, API and UI

- The FastAPI backend (`api/main.py`) and the UI (`ui/index.html`) work end to end. Trials on student-test trips with the tuned model and default HMM parameters gave these Top-1 accuracies:

| Trip | Classifier only | Classifier + HMM |
|---|---|---|
| 00000000 | 95.1% | 96.1% |
| 00000056 | 92.1% | 96.0% |
| 00000091 | 88.8% | 92.9% |

- HMM decoder tuning and the full classifier-vs-HMM comparison (`scripts/05_hmm.py`) are still running and will be in the final submission.
