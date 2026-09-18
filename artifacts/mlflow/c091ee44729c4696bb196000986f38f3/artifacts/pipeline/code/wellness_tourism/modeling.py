"""Leakage-safe classifier construction and threshold/evaluation functions."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from wellness_tourism.schema import CATEGORICAL, FEATURES, FeatureSelector


def build_pipeline(stage, estimator):
    categorical = [column for column in FEATURES[stage] if column in CATEGORICAL]
    numeric = [column for column in FEATURES[stage] if column not in CATEGORICAL]
    encoder = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("categorical", encoder, categorical),
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
    ])
    return Pipeline([("select", FeatureSelector(stage)), ("preprocess", preprocess), ("model", estimator)])


def choose_threshold(target, probabilities):
    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    scores = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(precision), where=(precision + recall) != 0)
    # Thresholds are ascending; the first F1 tie prioritizes recall.
    return float(thresholds[int(np.argmax(scores[:-1]))])


def evaluate(target, probabilities, threshold):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "confusion_matrix": confusion_matrix(target, predictions, labels=[0, 1]).tolist(),
    }