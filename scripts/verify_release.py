"""Verify newly trained artifacts, including container path portability."""

import argparse
import json
from pathlib import Path

import pandas as pd

from wellness_tourism.predict import load_registered_model, predict_customers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", action="store_true")
    parser.parse_args()
    latest = json.loads(Path("artifacts/latest.json").read_text())
    run_dir = Path("artifacts/runs") / Path(latest["run_dir"]).name
    registry = json.loads((run_dir / "registry.json").read_text())
    if registry["smoke"] or not registry["passed_gate"]:
        raise ValueError("The candidate release has not passed its gates.")
    for stage, entry in registry["models"].items():
        entry["path"] = str(run_dir / entry["asset"])
        model = load_registered_model(registry, stage)
        sample = pd.read_csv("data/test.csv").head(3)
        result = predict_customers(model, sample, registry, stage)
        assert len(result) == 3
        print(f"{stage}: valid registered model, probabilities {result.purchase_probability.round(4).tolist()}")


if __name__ == "__main__":
    main()