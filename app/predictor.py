"""Prediction service: loads the active models once and scores transactions.

Two scoring paths give identical results:
  * real-time path (up to 50 transactions): rebuilds user features from those users' own
    history and takes merchant profiles from a stored table, so a request is answered quickly;
  * batch path (larger files): rebuilds every feature from the full history in one pass.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app import model_components  # noqa: F401  (makes the TextGatedHybrid class importable)
from app.features import RAW_COLUMNS, build_features

PROFILE_SOURCES = ["log_amount", "hour", "is_weekend", "day_of_month", "is_night", "log_amount_vs_user", "is_month_start"]
STORE_COLUMNS = ["transaction_id", "transaction_date", "merchant"] + PROFILE_SOURCES
REALTIME_LIMIT = 50
ORDER = ["transaction_date", "transaction_id"]


class TransactionScorer:
    def __init__(self, project_root, history_file="data/processed/transaction_history.csv"):
        self.root = Path(project_root)
        registry = self.root / "models" / "registry"
        self.active = json.loads((registry / "active.json").read_text())
        clf_dir = registry / "category_classifier" / self.active["category_classifier"]
        det_dir = registry / "anomaly_detector" / self.active["anomaly_detector"]
        self.classifier = joblib.load(clf_dir / "model.joblib")
        self.detector = joblib.load(det_dir / "model.joblib")
        self.clf_features = json.loads((clf_dir / "metadata.json").read_text())["features"]
        self.det_features = self.detector["features"]
        history = pd.read_csv(self.root / history_file, parse_dates=["transaction_date"])
        self.raw_history = history[RAW_COLUMNS + ["category"]].sort_values(ORDER).reset_index(drop=True)
        self.raw_history["user_id"] = self.raw_history["user_id"].astype(object)
        self.raw_history["transaction_id"] = self.raw_history["transaction_id"].astype(object)
        self.known_ids = set(self.raw_history["transaction_id"])
        self.store = build_features(self.raw_history)[STORE_COLUMNS]   # inputs to merchant profiles

    # ---------- feature assembly ----------
    @staticmethod
    def _prepare(transactions):
        df = pd.DataFrame(transactions).copy()
        missing = [c for c in RAW_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"missing fields: {missing}")
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        df["amount"] = df["amount"].astype(float)
        if "category" not in df.columns:
            df["category"] = np.nan
        return df[RAW_COLUMNS + ["category"]].sort_values(ORDER).reset_index(drop=True)

    def _merchant_profiles(self, feats):
        """Merchant profiles use ALL earlier transactions at the merchant, by any user."""
        feats = feats.copy()
        for merchant, idx in feats.groupby("merchant").groups.items():
            m = pd.concat([self.store[self.store["merchant"] == merchant], feats.loc[idx, STORE_COLUMNS]])
            m = m.sort_values(ORDER).set_index("transaction_id")
            prior = m[PROFILE_SOURCES].shift(1)
            ids = feats.loc[idx, "transaction_id"]
            for col in PROFILE_SOURCES:
                feats.loc[idx, f"merchant_avg_{col}"] = prior[col].expanding().mean().loc[ids].to_numpy()
            feats.loc[idx, "merchant_sd_log_amount"] = prior["log_amount"].expanding().std().loc[ids].to_numpy()
        return feats

    def _features(self, new):
        ids = set(new["transaction_id"])
        if len(new) <= REALTIME_LIMIT:
            users = set(new["user_id"])
            own = self.raw_history[[u in users for u in self.raw_history["user_id"]]]
            feats = build_features(pd.concat([own, new], ignore_index=True))
            feats = feats[feats["transaction_id"].isin(ids)].reset_index(drop=True)
            return self._merchant_profiles(feats), "real-time"
        feats = build_features(pd.concat([self.raw_history, new], ignore_index=True))
        return feats[feats["transaction_id"].isin(ids)].reset_index(drop=True), "batch"

    def _log_amount_ratio(self, scored):
        """Amount relative to the user's earlier median in the same category (at least 5 earlier)."""
        users = set(scored["user_id"])
        frame = pd.concat([self.raw_history[[u in users for u in self.raw_history["user_id"]]],
                           scored[RAW_COLUMNS + ["category"]]], ignore_index=True).sort_values(ORDER)
        med = frame.groupby(["user_id", "category"])["amount"].transform(
            lambda s: s.shift(1).expanding(min_periods=5).median())
        ratio = (frame["amount"].abs() / med).fillna(1.0)
        return np.log(ratio.set_axis(frame["transaction_id"]).loc[scored["transaction_id"]].to_numpy())

    # ---------- public method ----------
    def score(self, transactions, update_history=False):
        t0 = time.perf_counter()
        new = self._prepare(transactions)
        if new["transaction_id"].duplicated().any() or not self.known_ids.isdisjoint(new["transaction_id"]):
            raise ValueError("transaction_id values must be new and unique")
        feats, path = self._features(new)
        new = new.set_index("transaction_id").loc[feats["transaction_id"]].reset_index()   # same order as feats

        # 1. Spending category
        #    Each transaction runs only the model its route needs: the text model when the description
        #    has known words, the 500-tree behavior model otherwise (same result as predict_proba).
        X = feats[self.clf_features]
        text_route = self.classifier.known_words(X)
        proba = np.zeros((len(X), len(self.classifier.classes_)))
        if text_route.any():
            proba[text_route] = self.classifier.text_model_.predict_proba(X[text_route])
        if (~text_route).any():
            proba[~text_route] = self.classifier.behavior_model_.predict_proba(X[~text_route])
        pred = self.classifier.classes_[proba.argmax(axis=1)]

        # 2. Anomaly score. The amount ratio needs a category: the supplied one if the bank already
        #    knows it, otherwise the category just predicted.
        new["category"] = np.where(new["category"].isna(), pred, new["category"])
        feats["log_amount_ratio"] = self._log_amount_ratio(new)
        pcts = []
        for d in self.detector["detectors"].values():
            s = -d["model"].score_samples(d["scaler"].transform(feats[self.det_features]))
            pcts.append(np.searchsorted(d["train_scores"], s) / len(d["train_scores"]))
        anomaly_score = np.mean(pcts, axis=0)

        result = pd.DataFrame({
            "transaction_id": feats["transaction_id"], "user_id": feats["user_id"],
            "merchant": feats["merchant"], "amount": feats["amount"],
            "predicted_category": pred, "confidence": proba.max(axis=1).round(4),
            "route": np.where(text_route, "text model", "behavior model"),
            "anomaly_score": anomaly_score.round(4),
            "is_anomaly": anomaly_score > self.detector["ensemble_threshold"],
            "classifier_version": self.active["category_classifier"],
            "detector_version": self.active["anomaly_detector"], "scoring_path": path})

        if update_history:   # streaming use: later transactions see this one as history
            self.known_ids.update(new["transaction_id"])
            self.raw_history = pd.concat([self.raw_history, new[RAW_COLUMNS + ["category"]]],
                                         ignore_index=True).sort_values(ORDER).reset_index(drop=True)
            self.store = pd.concat([self.store, feats[STORE_COLUMNS]], ignore_index=True).sort_values(ORDER)
        result.attrs["latency_ms"] = (time.perf_counter() - t0) * 1000
        return result
