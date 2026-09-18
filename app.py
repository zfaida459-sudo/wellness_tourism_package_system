"""Streamlit entrypoint for local execution and Community Cloud."""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from wellness_tourism.predict import load_registered_model, load_registry, predict_customers
from wellness_tourism.schema import FEATURES

st.set_page_config(page_title="Visit with Us | Purchase Propensity", page_icon=":material/travel_explore:", layout="wide")
st.html("""<style>
h1 {font-size: 2rem !important; letter-spacing: 0 !important;}
h3 {font-size: 1.125rem !important; letter-spacing: 0 !important;}
[data-testid="stCaptionContainer"] {overflow-wrap: anywhere;}
.block-container {padding-top: 2.5rem;}
</style>""")
st.title("Visit with Us")
st.subheader("Tourism purchase propensity")

stage = st.radio("Customer stage", ["pre_contact", "post_interaction"], horizontal=True,
                 format_func=lambda value: value.replace("_", " ").title())


@st.cache_resource(show_spinner="Loading registered model")
def cached_model(registry_text, selected_stage):
    return load_registered_model(json.loads(registry_text), selected_stage)


try:
    registry_path = Path(os.environ.get("WELLNESS_REGISTRY", "models/registry.json"))
    registry = load_registry(registry_path)
    model = cached_model(json.dumps(registry, sort_keys=True), stage)
except Exception as error:
    st.error(f"Model unavailable: {error}")
    st.stop()

st.caption(f"Model: {registry['models'][stage]['algorithm']} | Release: {registry['release']}")
single_tab, batch_tab, evaluation_tab = st.tabs(["Customer", "Batch", "Evaluation"])
options = {
    "CityTier": [1, 2, 3], "Occupation": ["Salaried", "Small Business", "Large Business", "Free Lancer"],
    "Gender": ["Female", "Male"], "PreferredPropertyStar": [3, 4, 5],
    "MaritalStatus": ["Single", "Married", "Divorced", "Unmarried"],
    "Designation": ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
    "TypeofContact": ["Self Enquiry", "Company Invited"],
    "ProductPitched": ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],
    "PitchSatisfactionScore": [1, 2, 3, 4, 5],
}
labels = {
    "Age": "Age", "CityTier": "City tier", "Occupation": "Occupation", "Gender": "Gender",
    "NumberOfPersonVisiting": "Party size", "PreferredPropertyStar": "Preferred hotel stars",
    "MaritalStatus": "Marital status", "NumberOfTrips": "Annual trips", "Passport": "Passport",
    "OwnCar": "Owns a car", "NumberOfChildrenVisiting": "Children travelling",
    "Designation": "Designation", "MonthlyIncome": "Monthly income",
    "TypeofContact": "Contact method", "DurationOfPitch": "Pitch duration",
    "NumberOfFollowups": "Follow-ups", "ProductPitched": "Product pitched",
    "PitchSatisfactionScore": "Pitch satisfaction",
}
defaults = {"Age": 35, "NumberOfPersonVisiting": 3, "NumberOfTrips": 2,
            "NumberOfChildrenVisiting": 1, "MonthlyIncome": 23000,
            "DurationOfPitch": 15, "NumberOfFollowups": 3}

with single_tab:
    with st.form(f"customer-{stage}"):
        columns = st.columns(3)
        customer = {}
        for position, field in enumerate(FEATURES[stage]):
            with columns[position % 3]:
                if field in ["Passport", "OwnCar"]:
                    customer[field] = int(st.checkbox(labels[field], key=f"{stage}-{field}"))
                elif field in options:
                    customer[field] = st.selectbox(labels[field], options[field], key=f"{stage}-{field}")
                else:
                    minimum = 1 if field in ["Age", "NumberOfPersonVisiting"] else 0
                    customer[field] = st.number_input(labels[field], min_value=minimum,
                                                       value=defaults[field], step=1, key=f"{stage}-{field}")
        submitted = st.form_submit_button("Score customer", icon=":material/query_stats:", type="primary")
    if submitted:
        try:
            result = predict_customers(model, pd.DataFrame([customer]), registry, stage).iloc[0]
            probability_column, decision_column = st.columns(2)
            probability_column.metric("Estimated purchase probability", f"{result.purchase_probability:.1%}")
            decision_column.metric("Prediction", "Purchase" if result.prediction else "No purchase")
            st.caption(f"Decision threshold: {registry['models'][stage]['threshold']:.3f}")
        except ValueError as error:
            st.error(str(error))

with batch_tab:
    st.download_button("CSV template", pd.DataFrame(columns=FEATURES[stage]).to_csv(index=False),
                       f"{stage}-template.csv", "text/csv", icon=":material/download:")
    uploaded = st.file_uploader("Customer CSV", type=["csv"])
    if uploaded is not None:
        try:
            if uploaded.size > 10 * 1024 * 1024:
                raise ValueError("CSV exceeds the 10 MB limit.")
            batch = pd.read_csv(uploaded, nrows=10001)
            st.dataframe(batch.head(20), use_container_width=True)
            if st.button("Score batch", icon=":material/query_stats:"):
                results = predict_customers(model, batch, registry, stage)
                st.dataframe(results, use_container_width=True)
                st.download_button("Download predictions", results.to_csv(index=False),
                                   "predictions.csv", "text/csv", icon=":material/download:")
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(str(error))

with evaluation_tab:
    metrics = registry["models"][stage]["test_metrics"]
    comparison = pd.DataFrame({"Metric": ["precision", "recall", "f1", "roc_auc", "average_precision"],
                               "Score": [metrics[name] for name in ["precision", "recall", "f1", "roc_auc", "average_precision"]]})
    st.bar_chart(comparison, x="Metric", y="Score", horizontal=True, color="#16816a")
    st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=["Actual no", "Actual yes"],
                              columns=["Predicted no", "Predicted yes"]), use_container_width=True)
    st.caption("Historical tourism purchases; wellness-specific and temporal validity are unverified.")