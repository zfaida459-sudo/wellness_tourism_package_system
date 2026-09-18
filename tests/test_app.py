import json
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from streamlit.testing.v1 import AppTest

from wellness_tourism.modeling import build_pipeline, evaluate
from wellness_tourism.predict import file_digest, load_registered_model, predict_customers
from wellness_tourism.schema import FEATURES


@pytest.fixture
def registry(tmp_path, monkeypatch):
    frame = pd.read_csv("data/tourism.csv").head(150)
    target = frame.pop("ProdTaken")
    registry = {"schema_version": 1, "release": "model-test", "passed_gate": True,
                "smoke": False, "versions": {"scikit-learn": version("scikit-learn")}, "models": {}}
    for stage in FEATURES:
        model = build_pipeline(stage, RandomForestClassifier(n_estimators=5, random_state=42))
        model.fit(frame, target)
        path = tmp_path / f"{stage}.joblib"
        joblib.dump(model, path)
        registry["models"][stage] = {"path": str(path), "sha256": file_digest(path),
                                     "features": FEATURES[stage], "threshold": 0.5,
                                     "algorithm": "test_random_forest",
                                     "test_metrics": evaluate(target, model.predict_proba(frame)[:, 1], 0.5)}
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry))
    monkeypatch.setenv("WELLNESS_REGISTRY", str(registry_path))
    return registry


def test_single_and_batch_match(registry):
    frame = pd.read_csv("data/tourism.csv").head(4)
    for stage in registry["models"]:
        model = load_registered_model(registry, stage)
        batch = predict_customers(model, frame, registry, stage)
        single = predict_customers(model, frame.iloc[[0]], registry, stage)
        assert batch.CustomerID.tolist() == frame.CustomerID.tolist()
        np.testing.assert_allclose(batch.purchase_probability.iloc[0], single.purchase_probability.iloc[0])
        assert batch.purchase_probability.between(0, 1).all()


def test_checksum_blocks_deserialization(registry):
    changed = json.loads(json.dumps(registry))
    changed["models"]["pre_contact"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        load_registered_model(changed, "pre_contact")


def test_app_stages_and_prediction(registry):
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 2
    app.radio[0].set_value("post_interaction").run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 2


def test_app_without_model(monkeypatch):
    monkeypatch.setenv("WELLNESS_REGISTRY", "does-not-exist.json")
    app = AppTest.from_file("app.py").run()
    assert not app.exception
    assert "Model unavailable" in app.error[0].value