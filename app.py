"""Interactive probability-of-default scorer."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import MODEL_PATH
from src.predict import load_artifact, score_applicant

st.set_page_config(page_title="Credit Risk Scorer", layout="centered")
st.title("Credit default scorer")
st.caption("Estimates probability of default from applicant and loan attributes.")

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

artifact = load_artifact()

with st.form("applicant"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=29)
        income = st.number_input("Annual income", min_value=1000, max_value=2_000_000, value=62000, step=1000)
        emp_length = st.number_input("Employment length (years)", min_value=0.0, max_value=50.0, value=4.0, step=0.5)
        home = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        prior_default = st.selectbox("Prior default on file", ["N", "Y"])
        cred_hist = st.number_input("Credit history length (years)", min_value=0, max_value=40, value=5)
    with col2:
        intent = st.selectbox(
            "Loan purpose",
            ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        )
        grade = st.selectbox("Loan grade", list("ABCDEFG"))
        amount = st.number_input("Loan amount", min_value=500, max_value=50000, value=10000, step=500)
        rate = st.number_input("Interest rate (%)", min_value=0.0, max_value=40.0, value=11.5, step=0.1)
        percent_income = amount / income if income else 0.0
        st.metric("Loan / income", f"{percent_income:.0%}")
    submitted = st.form_submit_button("Score applicant")

if submitted:
    result = score_applicant(
        {
            "person_age": age,
            "person_income": income,
            "person_home_ownership": home,
            "person_emp_length": emp_length,
            "loan_intent": intent,
            "loan_grade": grade,
            "loan_amnt": amount,
            "loan_int_rate": rate,
            "loan_percent_income": percent_income,
            "cb_person_default_on_file": prior_default,
            "cb_person_cred_hist_length": cred_hist,
        },
        artifact=artifact,
    )
    st.subheader(f"P(default) = {result['probability_of_default']:.1%}")
    st.write(
        f"Risk band **{result['risk_band']}** · threshold {result['threshold']:.2f} · "
        f"model `{result['model_name']}`"
    )
    if result["predicted_default"]:
        st.warning("Flagged as default risk at the F2 operating point.")
    else:
        st.success("Below the default-risk threshold at the F2 operating point.")
