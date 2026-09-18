import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from wellness_tourism.modeling import build_pipeline, choose_threshold, evaluate
from wellness_tourism.schema import INTERACTION


def test_training_only_preprocessing_and_roundtrip(tmp_path):
    frame = pd.read_csv("data/tourism.csv").head(150)
    target = frame.pop("ProdTaken")
    frame = frame.drop(columns=["CustomerID", "Unnamed: 0"])
    model = build_pipeline("pre_contact", RandomForestClassifier(n_estimators=5, random_state=42))
    model.fit(frame, target)
    learned = model.named_steps["preprocess"].named_transformers_["numeric"].statistics_.copy()
    check = frame.head(2).copy()
    check.loc[:, "Age"] = np.nan
    check.loc[:, "Occupation"] = "Previously unseen"
    probabilities = model.predict_proba(check)
    assert np.isfinite(probabilities).all()
    np.testing.assert_array_equal(learned, model.named_steps["preprocess"].named_transformers_["numeric"].statistics_)
    check[INTERACTION] = "ignored"
    np.testing.assert_allclose(probabilities, model.predict_proba(check))
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(probabilities, joblib.load(path).predict_proba(check))


def test_threshold_and_metrics():
    target = [0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.4, 0.8]
    threshold = choose_threshold(target, probabilities)
    assert threshold == 0.4
    assert evaluate(target, probabilities, threshold)["f1"] == 1.0