"""Register the source snapshot and prepare reproducible group-exclusive CSVs."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from wellness_tourism.schema import FEATURES, TARGET, normalize_features


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def customer_groups(raw, features):
    """Union customers and identical predictor records before dropping identifiers."""
    parents = list(range(len(raw)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    fingerprints = pd.util.hash_pandas_object(features, index=False).astype(str)
    identifiers = raw.get("CustomerID", pd.Series(np.nan, index=raw.index))
    for position, (identifier, fingerprint) in enumerate(zip(identifiers, fingerprints)):
        keys = [("record", fingerprint)]
        if pd.notna(identifier):
            keys.append(("customer", str(identifier)))
        for key in keys:
            if key in seen:
                parents[root(position)] = root(seen[key])
            else:
                seen[key] = position
    return np.asarray([root(position) for position in range(len(raw))])


def prepare_dataset(config):
    source = Path(config["data_path"])
    destination = Path(config["prepared_dir"])
    raw = pd.read_csv(source)
    if TARGET not in raw or raw[TARGET].isna().any() or not raw[TARGET].isin([0, 1]).all():
        raise ValueError("ProdTaken must contain only non-missing 0/1 labels.")
    features = normalize_features(raw, "post_interaction")
    groups = customer_groups(raw, features)
    cleaned = features.assign(**{TARGET: raw[TARGET].astype(int)})
    keep = ~cleaned.duplicated()
    cleaned = cleaned.loc[keep].reset_index(drop=True)
    groups = groups[keep]
    if cleaned[TARGET].nunique() != 2 or len(np.unique(groups)) < 5:
        raise ValueError("At least five independent groups and both target classes are required.")
    folds = np.full(len(cleaned), -1)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config["seed"])
    for fold, (_, indices) in enumerate(splitter.split(cleaned, cleaned[TARGET], groups)):
        folds[indices] = fold
        if cleaned.iloc[indices][TARGET].nunique() != 2:
            raise ValueError("Every partition must contain both classes; more independent data is needed.")
    destination.mkdir(parents=True, exist_ok=True)
    development = folds != 0
    cleaned.loc[development].to_csv(destination / "train.csv", index=False)
    cleaned.loc[~development].to_csv(destination / "test.csv", index=False)
    split_metadata = {
        "seed": config["seed"], "test_fold": 0, "validation_fold": 1,
        "train_groups": groups[development].tolist(), "train_folds": folds[development].tolist(),
        "test_groups": groups[~development].tolist(),
    }
    write_json(destination / "splits.json", split_metadata)
    manifest = {
        "source_file": source.as_posix(), "source_sha256": sha256(source),
        "source_revision": source_revision(), "rows_raw": len(raw), "rows_clean": len(cleaned),
        "duplicates_removed": int((~keep).sum()), "independent_groups": len(np.unique(groups)),
        "missing_values": {column: int(count) for column, count in features.isna().sum().items()},
        "purchase_rate": float(cleaned[TARGET].mean()), "features": FEATURES,
        "source_note": "User-provided course dataset; verify original redistribution terms before publishing.",
        "files": {name: sha256(destination / name) for name in ["train.csv", "test.csv", "splits.json"]},
        "rows_train": int(development.sum()), "rows_test": int((~development).sum()),
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def load_prepared(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for filename, digest in manifest["files"].items():
        if sha256(directory / filename) != digest:
            raise ValueError(f"Prepared data checksum mismatch: {filename}")
    train = pd.read_csv(directory / "train.csv")
    test = pd.read_csv(directory / "test.csv")
    splits = json.loads((directory / "splits.json").read_text(encoding="utf-8"))
    if set(splits["train_groups"]) & set(splits["test_groups"]):
        raise ValueError("Customer groups overlap between training and test data.")
    return train, test, splits, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    arguments = parser.parse_args()
    print(json.dumps(prepare_dataset(json.loads(Path(arguments.config).read_text())), indent=2))


if __name__ == "__main__":
    main()