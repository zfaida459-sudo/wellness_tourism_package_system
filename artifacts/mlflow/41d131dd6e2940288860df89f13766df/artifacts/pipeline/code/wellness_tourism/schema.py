"""Raw-input contracts shared by training, notebooks, and the app."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

TARGET = "ProdTaken"
PRE_CONTACT = [
    "Age", "CityTier", "Occupation", "Gender", "NumberOfPersonVisiting",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
INTERACTION = [
    "TypeofContact", "DurationOfPitch", "NumberOfFollowups", "ProductPitched",
    "PitchSatisfactionScore",
]
FEATURES = {"pre_contact": PRE_CONTACT, "post_interaction": PRE_CONTACT + INTERACTION}
CATEGORICAL = [
    "CityTier", "Occupation", "Gender", "MaritalStatus", "Designation",
    "TypeofContact", "ProductPitched",
]
TEXT = [column for column in CATEGORICAL if column != "CityTier"]
DISCRETE = {
    "CityTier": {1, 2, 3}, "Passport": {0, 1}, "OwnCar": {0, 1},
    "PreferredPropertyStar": {3, 4, 5}, "PitchSatisfactionScore": {1, 2, 3, 4, 5},
}
COUNTS = [
    "NumberOfPersonVisiting", "NumberOfChildrenVisiting", "NumberOfTrips",
    "NumberOfFollowups",
]


def normalize_features(frame, stage):
    """Select named raw fields and validate values without learning statistics."""
    if stage not in FEATURES:
        raise ValueError(f"Unknown prediction stage: {stage}")
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Input must be a DataFrame with unique column names.")
    missing = sorted(set(FEATURES[stage]) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    result = frame.loc[:, FEATURES[stage]].copy()
    for column in result:
        values = result[column].replace(r"^\s*$", np.nan, regex=True)
        if column in TEXT:
            result[column] = values.map(
                lambda value: str(value).strip() if pd.notna(value) else np.nan
            ).astype(object)
            if column == "Gender":
                result[column] = result[column].replace({"Fe Male": "Female"})
            continue
        numeric = pd.to_numeric(values, errors="coerce").astype(float)
        invalid = (values.notna() & numeric.isna()) | (numeric.notna() & ~np.isfinite(numeric))
        invalid |= numeric < 0
        if column in DISCRETE:
            invalid |= numeric.notna() & ~numeric.isin(DISCRETE[column])
        if column in COUNTS:
            invalid |= numeric.notna() & (numeric % 1 != 0)
        if column == "Age":
            invalid |= (numeric <= 0) | (numeric > 120)
        if column == "NumberOfPersonVisiting":
            invalid |= numeric < 1
        if invalid.any():
            rows = result.index[invalid].tolist()[:5]
            raise ValueError(f"Invalid values for {column} at rows {rows}.")
        result[column] = numeric
    if {"NumberOfChildrenVisiting", "NumberOfPersonVisiting"} <= set(result):
        invalid = result["NumberOfChildrenVisiting"] > result["NumberOfPersonVisiting"]
        if invalid.any():
            raise ValueError(f"Children exceed party size at rows {result.index[invalid].tolist()[:5]}.")
    return result


class FeatureSelector(TransformerMixin, BaseEstimator):
    """An importable selector ensures serialized models keep their input contract."""

    def __init__(self, stage="pre_contact"):
        self.stage = stage

    def fit(self, frame, target=None):
        normalize_features(frame, self.stage)
        self.n_features_in_ = len(frame.columns)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, frame):
        check_is_fitted(self)
        return normalize_features(frame, self.stage)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES[self.stage], dtype=object)