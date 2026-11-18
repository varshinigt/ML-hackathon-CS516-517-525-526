"""Write artifacts/pipeline_config.json pointing the API at the final model + HMM config."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.features import DEFAULT_CFG, FEATURES  # noqa: E402
from pathsnap.io import ROOT  # noqa: E402

name = sys.argv[1]
models = sorted((ROOT / "models").glob(f"final_{name}_*.joblib"))
if not models:
    sys.exit(f"no final model for {name}; run scripts/05_hmm.py --model {name} first")
cfg = {
    "model_name": f"{name} ({models[0].stem.split('_')[-1]})",
    "model_path": str(models[0].relative_to(ROOT)),
    "hmm_config": "artifacts/hmm_config.json",
    "features": FEATURES,
    "candidates": DEFAULT_CFG,
    "trained_on": "80 training trips (data/raw/train)",
}
(ROOT / "artifacts" / "pipeline_config.json").write_text(json.dumps(cfg, indent=2))
print(json.dumps(cfg, indent=2))
