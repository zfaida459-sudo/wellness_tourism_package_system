import json

import numpy as np
import pandas as pd
import pytest

from wellness_tourism.data import customer_groups, load_prepared, prepare_dataset
from wellness_tourism.schema import FEATURES, INTERACTION, normalize_features


def test_pre_contact_excludes_sales_and_eda():
    original = pd.read_csv("data/tourism.csv").head(5)
    changed = original.copy()
    changed[INTERACTION] = "not available"
    changed["IncomeCategory"] = "EDA only"
    pd.testing.assert_frame_equal(
        normalize_features(original, "pre_contact"), normalize_features(changed, "pre_contact")
    )
    assert len(FEATURES["pre_contact"]) == 13
    assert len(FEATURES["post_interaction"]) == 18


def test_values_and_missing_columns():
    frame = pd.read_csv("data/tourism.csv").head(2)
    with pytest.raises(ValueError, match="Missing required"):
        normalize_features(frame.drop(columns="Age"), "pre_contact")
    frame.loc[0, "Age"] = np.inf
    with pytest.raises(ValueError, match="Age"):
        normalize_features(frame, "pre_contact")
    frame.loc[0, "Age"] = np.nan
    assert np.isnan(normalize_features(frame, "pre_contact").iloc[0]["Age"])


def test_grouping_merges_customer_and_duplicate_links():
    raw = pd.DataFrame({"CustomerID": [1, 2, 2, 3]})
    features = pd.DataFrame({"value": [5, 5, 7, 8]})
    groups = customer_groups(raw, features)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_preparation_is_reproducible_and_disjoint(tmp_path):
    config = json.loads(open("config.json").read())
    config["prepared_dir"] = str(tmp_path)
    first = prepare_dataset(config)
    train, test, splits, _ = load_prepared(tmp_path)
    second = prepare_dataset(config)
    assert first["files"] == second["files"]
    assert len(train) + len(test) == first["rows_clean"]
    assert not set(splits["train_groups"]) & set(splits["test_groups"])
    assert set(train.ProdTaken) == set(test.ProdTaken) == {0, 1}
    assert "CustomerID" not in train
    (tmp_path / "test.csv").write_text("modified")
    with pytest.raises(ValueError, match="checksum"):
        load_prepared(tmp_path)