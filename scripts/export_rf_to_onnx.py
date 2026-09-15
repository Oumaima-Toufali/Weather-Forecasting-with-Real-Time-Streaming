# Path: scripts/export_rf_to_onnx.py
"""
Export trained RandomForest models (pickle) to ONNX format.

This is optional and does not change the serving pipeline. It simply
creates ONNX artifacts in the `models/` folder.

Usage:
    python scripts/export_rf_to_onnx.py

Requirements:
    pip install skl2onnx
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

try:
    import joblib
    import numpy as np
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except ImportError:
    print(
        "[ERROR] Missing dependency 'skl2onnx'. "
        "Install with: pip install skl2onnx"
    )
    sys.exit(1)


MODELS_DIR = Path("models")
METADATA_PATH = MODELS_DIR / "model_metadata.json"
OUTPUT_MANIFEST = MODELS_DIR / "onnx_manifest.json"
HORIZONS = [1, 3, 6]


def load_metadata() -> List[str]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata: Dict = json.load(f)
    feature_columns = metadata.get("feature_columns")
    if not feature_columns:
        raise ValueError("No feature_columns in model_metadata.json")
    return feature_columns


def export_model(horizon: int, feature_columns: List[str]) -> Path:
    model_path = MODELS_DIR / f"weather_model_{horizon}h.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    # Build a dummy input to define the ONNX graph shape
    n_features = len(feature_columns)
    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    onnx_model = convert_sklearn(model, initial_types=initial_type)

    output_path = MODELS_DIR / f"weather_model_{horizon}h.onnx"
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return output_path


def main() -> None:
    print("=== Export RandomForest models to ONNX ===")
    feature_columns = load_metadata()
    manifest = {}

    for h in HORIZONS:
        try:
            path = export_model(h, feature_columns)
            manifest[f"{h}h"] = {"path": str(path), "features": feature_columns}
            print(f"[OK] Exported horizon {h}h -> {path}")
        except Exception as exc:
            print(f"[WARN] Could not export horizon {h}h: {exc}")

    if manifest:
        with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[OK] Manifest written: {OUTPUT_MANIFEST}")
    else:
        print("[ERROR] No ONNX models exported.")


if __name__ == "__main__":
    main()


