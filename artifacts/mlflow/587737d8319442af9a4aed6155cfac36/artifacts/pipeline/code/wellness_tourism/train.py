"""Tune two-stage classifiers, track experiments, and export verified releases."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import uuid

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from xgboost import XGBClassifier

from wellness_tourism.data import load_prepared, source_revision, write_json, sha256
from wellness_tourism.modeling import build_pipeline, choose_threshold, evaluate
from wellness_tourism.schema import FEATURES, TARGET, TEXT, normalize_features


def export_plots(target, probabilities, destination):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    PrecisionRecallDisplay.from_predictions(target, probabilities, ax=axes[0])
    RocCurveDisplay.from_predictions(target, probabilities, ax=axes[1])
    figure.tight_layout()
    figure.savefig(destination, dpi=120)
    plt.close(figure)


def run_training(config, smoke=False, registry_path="models/registry.json"):
    train, test, splits, data_manifest = load_prepared(config["prepared_dir"])
    fit_mask = np.asarray(splits["train_folds"]) != 1
    features = train.drop(columns=TARGET)
    fit_features, validation = features.loc[fit_mask], features.loc[~fit_mask]
    fit_target, validation_target = train.loc[fit_mask, TARGET], train.loc[~fit_mask, TARGET]
    groups = np.asarray(splits["train_groups"])[fit_mask]
    cv = list(StratifiedGroupKFold(
        n_splits=config["cv_folds"], shuffle=True, random_state=config["seed"]
    ).split(fit_features, fit_target, groups))
    for fit_indices, check_indices in cv:
        if fit_target.iloc[fit_indices].nunique() != 2 or fit_target.iloc[check_indices].nunique() != 2:
            raise ValueError("Each training CV fold must contain both target classes.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = Path(config["artifacts_dir"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    tracking_root = Path(config["artifacts_dir"]).parent.resolve()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{tracking_root / 'mlflow.db'}"))
    experiment_name = "wellness-tourism"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=(tracking_root / "mlflow").as_uri())
    mlflow.set_experiment(experiment_name)
    runtime_versions = {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib"]}
    report = {
        "run_id": run_id, "smoke": smoke, "source_revision": source_revision(),
        "python": platform.python_version(), "versions": runtime_versions, "config": config,
        "data": data_manifest, "stages": {},
        "limitations": [
            "Historical tourism purchases are a proxy, not verified wellness-package outcomes.",
            "No timestamps establish whether interaction fields were measured before purchase.",
            "Travel preferences are assumed available in customer profiles before contact.",
            "Estimated model probabilities are not guaranteed calibrated probabilities.",
        ],
    }
    registry = {"schema_version": 1, "release": f"model-{run_id}", "versions": runtime_versions,
                "smoke": smoke, "models": {}}
    with mlflow.start_run(run_name=run_id) as parent:
        report["mlflow_run_id"] = parent.info.run_id
        mlflow.log_params({"seed": config["seed"], "smoke": smoke, "data_sha256": data_manifest["source_sha256"]})
        mlflow.set_tag("source_revision", report["source_revision"])
        for stage in FEATURES:
            print(f"Tuning {stage}", flush=True)
            candidates = {
                "random_forest": RandomForestClassifier(random_state=config["seed"], n_jobs=1),
                "xgboost": XGBClassifier(random_state=config["seed"], n_jobs=1, tree_method="hist", eval_metric="logloss"),
            }
            searches = {}
            for name, estimator in candidates.items():
                grid = config[f"{name}_grid"]
                if smoke:
                    grid = {"model__n_estimators": [10], "model__max_depth": [3]}
                search = GridSearchCV(build_pipeline(stage, estimator), grid, scoring="average_precision",
                                      cv=cv, n_jobs=1, error_score="raise", return_train_score=True)
                search.fit(fit_features, fit_target)
                searches[name] = search
                results = pd.DataFrame(search.cv_results_)
                results.to_csv(output / f"{stage}-{name}-cv.csv", index=False)
                for position, parameters in enumerate(search.cv_results_["params"]):
                    with mlflow.start_run(run_name=f"{stage}-{name}-{position}", nested=True):
                        mlflow.log_params({"stage": stage, "algorithm": name, **parameters})
                        mlflow.log_metrics({
                            key: float(values[position]) for key, values in search.cv_results_.items()
                            if key.endswith("_score") or key in {"mean_fit_time", "std_fit_time"}
                        })
                print(f"  {name}: CV average precision {search.best_score_:.4f}", flush=True)
            selected_name = max(searches, key=lambda name: searches[name].best_score_)
            selected = searches[selected_name].best_estimator_
            validation_probabilities = selected.predict_proba(validation)[:, 1]
            threshold = choose_threshold(validation_target, validation_probabilities)
            dummy = build_pipeline(stage, DummyClassifier(strategy="prior"))
            dummy.fit(fit_features, fit_target)
            baseline = evaluate(validation_target, dummy.predict_proba(validation)[:, 1], 0.0)
            scores = {
                "train": evaluate(fit_target, selected.predict_proba(fit_features)[:, 1], threshold),
                "validation": evaluate(validation_target, validation_probabilities, threshold),
                "test": evaluate(test[TARGET], selected.predict_proba(test.drop(columns=TARGET))[:, 1], threshold),
                "dummy_validation": baseline,
            }
            model_file = output / f"{stage}.joblib"
            joblib.dump(selected, model_file, compress=3)
            restored = joblib.load(model_file)
            np.testing.assert_allclose(restored.predict_proba(validation), selected.predict_proba(validation))
            export_plots(test[TARGET], selected.predict_proba(test.drop(columns=TARGET))[:, 1], output / f"{stage}-curves.png")
            importance_input = normalize_features(validation, stage)
            importance = permutation_importance(selected[1:], importance_input, validation_target,
                                                scoring="average_precision", n_repeats=2, random_state=42)
            pd.DataFrame({"feature": importance_input.columns, "importance": importance.importances_mean}).sort_values(
                "importance", ascending=False
            ).to_csv(output / f"{stage}-importance.csv", index=False)
            stage_report = {
                "algorithm": selected_name, "best_params": searches[selected_name].best_params_,
                "cv_average_precision": float(searches[selected_name].best_score_),
                "threshold": threshold, "metrics": scores,
                "passed_gate": scores["validation"]["average_precision"] > baseline["average_precision"],
            }
            with mlflow.start_run(run_name=f"{stage}-selected", nested=True) as selected_run:
                mlflow.log_params({"stage": stage, "threshold": threshold, **stage_report["best_params"]})
                for partition, metrics in scores.items():
                    mlflow.log_metrics({f"{partition}_{key}": value for key, value in metrics.items() if key != "confusion_matrix"})
                signature = ModelSignature(inputs=Schema([
                    ColSpec("string" if field in TEXT else "double", field, required=False)
                    for field in FEATURES[stage]
                ]))
                mlflow.sklearn.log_model(
                    selected, artifact_path="pipeline", signature=signature,
                    pip_requirements=[f"{name}=={value}" for name, value in runtime_versions.items()],
                    code_paths=["wellness_tourism"],
                    registered_model_name=None if smoke else f"wellness_{stage}",
                )
                stage_report["mlflow_run_id"] = selected_run.info.run_id
            report["stages"][stage] = stage_report
            registry["models"][stage] = {
                "path": model_file.resolve().as_posix(), "asset": model_file.name,
                "sha256": sha256(model_file), "features": FEATURES[stage],
                "threshold": threshold, "algorithm": selected_name,
                "test_metrics": scores["test"],
            }
        report["passed_gate"] = all(value["passed_gate"] for value in report["stages"].values())
        registry["passed_gate"] = report["passed_gate"]
        write_json(output / "report.json", report)
        write_json(output / "registry.json", registry)
        for filename in ["manifest.json", "splits.json", "train.csv", "test.csv"]:
            shutil.copy2(Path(config["prepared_dir"]) / filename, output / filename)
        mlflow.log_artifacts(str(output), artifact_path="release")
    if not smoke:
        write_json("artifacts/latest.json", {"run_dir": output.resolve().as_posix()})
        if report["passed_gate"]:
            write_json(registry_path, registry)
    print(f"Run saved to {output}; validation gate: {report['passed_gate']}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    run_training(json.loads(Path(arguments.config).read_text()), smoke=arguments.smoke)


if __name__ == "__main__":
    main()