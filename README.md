Personal Finance Transaction Intelligence
![Open in Streamlit](https://pfti-ngaji-calvince.streamlit.app)
Live application: https://pfti-ngaji-calvince.streamlit.app
```math
\\begin{aligned}
\&\\textbf{Deployed machine learning system, model version 3.1.0} \\\\\[4pt]
\&\\text{Spending category:}\\quad \\hat{y} = \\arg\\max\_{k}\\; P\\left(\\text{category} = k \\mid \\text{description},\\ \\text{behaviour}\\right), \\qquad \\text{accuracy} = 0.944 \\\\\[4pt]
\&\\text{Unusual transactions:}\\quad s(x) = \\tfrac{1}{2}\\left\[\\,F\_{\\text{IF}}(x) + F\_{\\text{SVM}}(x)\\,\\right], \\quad \\text{flag if } s(x) > \\tau\_{99.5}, \\qquad F\_1 = 0.672
\\end{aligned}
```
Overview
An end-to-end machine learning project that predicts the spending category of a bank transaction and flags transactions that are unusual for the customer. It covers data cleaning, feature engineering, model training and refinement, packaging, a REST API, a web application, monitoring, retraining and live deployment. All data are simulated; no real customer data are used.
Models
Task	Model	Held-out test result
Spending category (8 classes)	Logistic regression on statement text, gated with extremely randomized trees on spending behaviour	Accuracy 0.944, F1 0.937
Unusual transactions	Average of isolation forest and one-class SVM percentile scores, alert above the 99.5th training percentile	Precision 0.821, recall 0.568, F1 0.672
What the live application does
Score a transaction: enter a user, date, time, description, merchant and amount, and get the predicted category, its confidence, the anomaly score and a review decision.
Score a file: upload a CSV of up to 5,000 transactions (or use the sample file) and download the scored results.
Model card: model descriptions, offline test metrics and file checksums.
Engineering
37 features rebuilt from raw transactions by one shared module, so training and serving use identical code.
Versioned model registry with metadata cards (`models/registry/`).
Real-time path (up to 50 transactions) and batch path give identical results.
Library versions pinned in `requirements.txt` to the versions the models were trained with.
Run locally
```bash
pip install -r requirements.txt
streamlit run app/streamlit\_app.py
```
Repository structure
Path	Contents
`app/streamlit\_app.py`	Web application (entry point)
`app/features.py`	Feature pipeline
`app/predictor.py`	Prediction service
`app/model\_components.py`	Custom model class
`models/registry/`	Model files and metadata
`data/`	Transaction history and sample file (simulated)
Tools
Python 3.14, scikit-learn 1.9.0, pandas, NumPy, Streamlit, Streamlit Community Cloud
Author
Ngaji Calvince Otieno
