# Personal Finance Transaction Intelligence

Web application that predicts the spending category of a bank transaction and flags transactions
that are unusual for the user. All data is simulated; no real customer data is included.

* Entrypoint: `app/streamlit_app.py`
* Models: `models/registry/` (category classifier and anomaly detector, version 3.1.0)
* Python 3.14, scikit-learn 1.9.0 (pinned in `requirements.txt`)

Run locally: `pip install -r requirements.txt` then `streamlit run app/streamlit_app.py`
