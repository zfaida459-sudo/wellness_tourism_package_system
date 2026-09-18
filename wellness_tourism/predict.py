"""Load only trusted, checksummed project models; never accept uploaded pickles."""

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import tempfile

import joblib
import numpy as np
import pandas as pd
import requests

from wellness_tourism.schema import FEATURES


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_registry(path="models/registry.json"):
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1 or set(registry.get("models", {})) != set(FEATURES):
        raise ValueError("Unsupported or incomplete model registry.")
    if not registry.get("passed_gate") or registry.get("smoke"):
        raise ValueError("This model release has not passed the publication gates.")
    return registry


def load_registered_model(registry, stage, cache_dir=".model-cache"):
    entry = registry["models"][stage]
    if entry["features"] != FEATURES[stage] or not 0 <= entry["threshold"] <= 1:
        raise ValueError("Model feature schema or threshold is invalid.")
    for package, expected in registry["versions"].items():
        if version(package) != expected:
            raise ValueError(f"Model needs {package}=={expected}; install the pinned requirements.")
    digest = entry["sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid model checksum.")
    if "path" in entry:
        path = Path(entry["path"])
    else:
        repository, release, asset = registry["repository"], registry["release"], entry["asset"]
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository) or not re.fullmatch(r"model-[\w.-]+", release):
            raise ValueError("Untrusted model repository or release identifier.")
        if asset != f"{stage}.joblib":
            raise ValueError("Unexpected release asset name.")
        url = f"https://github.com/{repository}/releases/download/{release}/{asset}"
        if entry.get("url") != url:
            raise ValueError("Model URL does not match the pinned GitHub release.")
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"{digest}.joblib"
        if not path.exists() or file_digest(path) != digest:
            temporary = None
            try:
                with requests.get(url, stream=True, timeout=(10, 60)) as response:
                    response.raise_for_status()
                    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
                        temporary = Path(stream.name)
                        size = 0
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            size += len(block)
                            if size > 100 * 1024 * 1024:
                                raise ValueError("Model download exceeds the 100 MB limit.")
                            stream.write(block)
                if file_digest(temporary) != digest:
                    raise ValueError("Downloaded model checksum mismatch.")
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    if file_digest(path) != digest:
        raise ValueError("Model checksum mismatch; refusing to deserialize.")
    model = joblib.load(path)
    if model.named_steps["select"].stage != stage or list(model.classes_) != [0, 1]:
        raise ValueError("Loaded model does not match the requested classification stage.")
    return model


def predict_customers(model, frame, registry, stage):
    if not 1 <= len(frame) <= 10000:
        raise ValueError("Provide between 1 and 10,000 customer rows.")
    probability = model.predict_proba(frame)[:, 1]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Model returned invalid probabilities.")
    result = pd.DataFrame({
        "stage": stage, "purchase_probability": probability,
        "prediction": (probability >= registry["models"][stage]["threshold"]).astype(int),
        "model_version": registry["release"],
    }, index=frame.index)
    if "CustomerID" in frame:
        result.insert(0, "CustomerID", frame["CustomerID"])
    return result