# PathSnap: map-matching and route reconstruction

PathSnap matches noisy GPS fixes to road segments and rebuilds a connected route.

For each GPS fix, the pipeline:

1. Generates the nearby candidate road arcs.
2. Scores every candidate with a trained classifier.
3. Decodes the most likely connected sequence of arcs with an HMM and the Viterbi algorithm.

A FastAPI backend serves the pipeline, and a browser UI lets the evaluator pick a trip and see the predicted route and its accuracy.

UE24CS352A Machine Learning hackathon. Team: CS516, CS517, CS525, CS526.

## Repository layout

| Folder | Contents |
|---|---|
| `src/pathsnap/` | Library code: data loading (`io.py`), geodesy (`geo.py`), road graph (`network.py`), cleaning and reference-label alignment (`preprocess.py`), candidates and features (`features.py`), model zoo (`models.py`), metrics (`metrics.py`), HMM/Viterbi (`hmm.py`), Folium maps (`viz.py`), inference pipeline (`pipeline.py`) |
| `scripts/` | Numbered experiment scripts, run in order (see below) |
| `api/` | FastAPI backend, run on Laptop 1 |
| `ui/` | Evaluator web UI (static HTML and Leaflet), run on Laptop 2 |
| `models/` | Saved models: every compared model, the tuned model and the final model |
| `artifacts/` | `pipeline_config.json` (which model the API loads) and `hmm_config.json` (decoder parameters) |
| `results/` | CSV results, experiment logs and JSON configs, figures (`figures/`), Folium maps (`maps/`), EDA tables (`eda/`) |
| `report/` | Technical report (`report.pdf`), its HTML source and the generator script |
| `data/raw/` | Raw trips go here (not committed, see Data) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Copy the organisers' folders into `data/raw/`, keeping the partition folder names:

```
data/raw/train/<trip>/<trip>.{track,nodes,arcs,route}          80 trips
data/raw/student_test/<trip>/<trip>.{track,nodes,arcs,route}   10 trips
data/raw/eval_test/<trip>/...                                  10 evaluator trips (only at evaluation time)
```

Partition membership is read from these folder names. Trips are never re-split.

## Reproducing the experiments

Run the scripts from the repository root, in this order:

```bash
python scripts/01_eda.py                       # dataset inventory, EDA plots, Folium maps
python scripts/02_build_features.py            # cleaning, label alignment, candidates and features (train + student_test)
python scripts/03_train_compare.py             # 10 models: 5-fold group CV + student-test evaluation
python scripts/04_bayes_opt.py --trials 30     # Optuna TPE on the model with the best CV Top-1
python scripts/05_hmm.py --model LightGBM      # decoder tuning (train OOF) + classifier vs HMM on student_test
python scripts/06_export.py LightGBM           # write artifacts/pipeline_config.json for the API
python report/build_report.py                  # regenerate report/report.html and report/report.pdf
```

The evaluator partition is never read by scripts 01–06. It is used only by the API at evaluation time.

## Running the system (two laptops)

**Laptop 1: API.** Put the 10 evaluator trips in `data/raw/eval_test/`, or point `PATHSNAP_TRIPS_DIR` at them, then run:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Every request is logged to the console and to `logs/api.log`. Find the laptop's IP address with `ipconfig` (Windows) or `ip addr` (Linux/macOS), and allow port 8000 through the firewall.

To rehearse with the student-test trips:

```bash
PATHSNAP_TRIPS_DIR=data/raw/student_test uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Laptop 2: UI.**

```bash
cd ui && python -m http.server 8080
```

Open `http://localhost:8080`, enter `http://<laptop-1-ip>:8000` as the API URL and click **Load trips**. Then select a trip. The UI shows the GPS fixes, the predicted route, the Top-1 accuracy and the route metrics.

The reference routes never leave Laptop 1. The API reads them only after prediction, to compute accuracy, and returns only the metrics.

### API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | none | Status and the loaded model |
| GET | `/trips` | none | Trip ids and GPS fix counts |
| POST | `/predict` | `{"trip_id": "00000056"}` | GPS fixes, predicted route polylines, the arc id for every observation, metrics, timings |

## Method summary

- **Distances.** The data uses lon/lat angles, so each trip is projected to a local tangent plane in metres before any geometry is computed.
- **Reference labels.** The `.route` files give the ordered arcs a vehicle travelled, not which arc each fix belongs to. Each fix's reference arc is found by a monotone dynamic-programming alignment of the fixes to the route. A label is treated as reliable when the fix lies within 50 m of its aligned arc.
- **Candidates.** Each fix gets up to 8 arcs within 60 m. If there are none, the search widens to 200 m, and then to the single nearest arc.
- **Features.** Each candidate is described by 19 features: distance and rank, heading agreement, position along the arc, speed and time gaps, the distances of neighbouring fixes, and connectivity to the neighbours' nearest arcs.
- **Models.** 10 models from 7 families are compared with GroupKFold cross-validation by trip. The model with the best CV Top-1 is tuned with Optuna's TPE sampler.
- **HMM.** The emissions are the classifier probabilities. The transitions follow Newson and Krumm: `-|network distance - straight-line distance| / beta`. Long time gaps cut the chain, and unreachable transitions get a finite penalty instead of making decoding fail.

## AI-assistance disclosure

Parts of this code and the report draft were written with an AI assistant. Every number, plot and map comes from running the code in this repository on the provided data. The team has reviewed the work and is responsible for explaining it.
