"""Streamlit web application for the Personal Finance Transaction Intelligence models.

Local run:   streamlit run app/streamlit_app.py
Cloud:       Streamlit Community Cloud, entrypoint app/streamlit_app.py
"""
import json
import sys
import uuid
from datetime import date, datetime, time as dtime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.features import RAW_COLUMNS          # noqa: E402
from app.predictor import TransactionScorer   # noqa: E402

st.set_page_config(page_title="Transaction Intelligence", page_icon=":material/receipt_long:", layout="wide")


# ---------- optional access control: set app_password in the app's Secrets to require a password ----------
def password_ok():
    try:
        expected = st.secrets.get("app_password")
    except Exception:          # no secrets file: open demo
        expected = None
    if not expected:
        return True
    if st.session_state.get("authenticated"):
        return True
    entered = st.text_input("Password", type="password")
    if entered and entered == expected:
        st.session_state["authenticated"] = True
        st.rerun()
    elif entered:
        st.error("Incorrect password")
    return False


@st.cache_resource(show_spinner="Loading models and transaction history...")
def load_scorer():
    return TransactionScorer(ROOT)


def model_cards(scorer):
    reg = ROOT / "models" / "registry"
    return {name: json.loads((reg / name / scorer.active[name] / "metadata.json").read_text())
            for name in ["category_classifier", "anomaly_detector"]}


def show_results(res):
    view = res[["transaction_id", "user_id", "merchant", "amount", "predicted_category", "confidence",
                "route", "anomaly_score", "is_anomaly"]].copy()
    view["is_anomaly"] = view["is_anomaly"].map({True: "Flagged", False: "Normal"})
    st.dataframe(view, hide_index=True, width="stretch")


if not password_ok():
    st.stop()

scorer = load_scorer()
cards = model_cards(scorer)
users = sorted(scorer.raw_history["user_id"].unique())

st.title("Personal Finance Transaction Intelligence")
st.caption("Predicts the spending category of a bank transaction and flags transactions that are unusual for the user. "
           "All data in this demonstration is simulated; no real customer data is used.")

with st.sidebar:
    st.subheader("Active models")
    st.write(f"Category classifier **v{scorer.active['category_classifier']}**")
    st.write(f"Anomaly detector **v{scorer.active['anomaly_detector']}**")
    st.write(f"History: **{len(scorer.raw_history):,}** transactions to "
             f"{scorer.raw_history['transaction_date'].max():%d %b %Y}")
    st.caption("Built with scikit-learn, FastAPI and Streamlit.")

tab_one, tab_file, tab_model = st.tabs(["Score a transaction", "Score a file", "Model card"])

with tab_one:
    with st.form("single"):
        c1, c2, c3 = st.columns(3)
        user = c1.selectbox("User", users, index=users.index("U008") if "U008" in users else 0)
        day = c2.date_input("Date", value=date(2026, 7, 3))
        at = c3.time_input("Time", value=dtime(3, 5))
        c4, c5, c6 = st.columns([2, 2, 1])
        description = c4.text_input("Statement description", value="VISA UBER", max_chars=120)
        merchant = c5.text_input("Merchant", value="Uber", max_chars=80)
        amount = c6.number_input("Amount (USD)", min_value=0.01, max_value=100000.0, value=41.75, step=1.0)
        submitted = st.form_submit_button("Score transaction", type="primary")
    if submitted:
        tx = {"transaction_id": f"WEB-{uuid.uuid4().hex[:10]}", "user_id": user,
              "transaction_date": datetime.combine(day, at), "description": description.strip(),
              "merchant": merchant.strip(), "amount": float(amount)}
        if not tx["description"] or not tx["merchant"]:
            st.error("Description and merchant are required.")
        else:
            res = scorer.score([tx])
            r = res.iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted category", r["predicted_category"].title(), f"confidence {r['confidence']:.0%}",
                      delta_color="off")
            m2.metric("Anomaly score", f"{r['anomaly_score']:.3f}",
                      f"alert above {scorer.detector['ensemble_threshold']:.3f}", delta_color="off")
            m3.metric("Decision", "Flag for review" if r["is_anomaly"] else "Normal", r["route"], delta_color="off")
            if r["is_anomaly"]:
                st.warning("This transaction is unusual for this user and would be sent for review.")
            if r["confidence"] < 0.5:
                st.info("Low confidence: this merchant has little or no history, so the customer should confirm the category.")
            st.caption(f"Scored in {res.attrs['latency_ms']:.0f} ms by the {r['scoring_path']} path.")

with tab_file:
    st.write("Upload a CSV with the columns `" + "`, `".join(RAW_COLUMNS) + "` (up to 5,000 rows), "
             "or use the sample file.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    use_sample = st.button("Use the sample file")
    data = None
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    elif use_sample:
        data = pd.read_csv(ROOT / "data" / "sample_transactions.csv")
    if data is not None:
        missing = [c for c in RAW_COLUMNS if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        elif len(data) > 5000:
            st.error("Please upload at most 5,000 rows.")
        elif data["transaction_id"].astype(str).isin(set(scorer.raw_history["transaction_id"])).any():
            st.error("Some transaction_id values already exist in the history. Use new IDs.")
        else:
            res = scorer.score(data[RAW_COLUMNS])
            a, b, c = st.columns(3)
            a.metric("Transactions scored", f"{len(res):,}")
            b.metric("Flagged as unusual", f"{int(res['is_anomaly'].sum()):,}")
            c.metric("Time taken", f"{res.attrs['latency_ms'] / 1000:.2f} s")
            show_results(res)
            st.download_button("Download results (CSV)", res.to_csv(index=False).encode(),
                               file_name="scored_transactions.csv", mime="text/csv")

with tab_model:
    clf, det = cards["category_classifier"], cards["anomaly_detector"]
    st.subheader("Category classifier")
    st.write(clf["notes"])
    st.dataframe(pd.DataFrame([clf["offline_test_metrics"]]), hide_index=True)
    st.subheader("Anomaly detector")
    st.write(det["notes"])
    st.dataframe(pd.DataFrame([det["offline_test_metrics"]]), hide_index=True)
    st.caption(f"Offline metrics are from the held-out test period (April to June 2026). "
               f"Model files: SHA-256 {clf['sha256'][:12]}... and {det['sha256'][:12]}..., "
               f"scikit-learn {clf['scikit_learn']}.")
