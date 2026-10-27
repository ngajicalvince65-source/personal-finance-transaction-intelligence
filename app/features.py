"""Feature pipeline for the Personal Finance Transaction Intelligence service.

Rebuilds every model feature from raw transaction fields (user_id, transaction_date,
description, merchant, amount, and category for past transactions), using exactly the
definitions from the earlier project stages. Every feature uses only transactions that
happened BEFORE the one being scored, so the same code serves training, batch scoring
and real-time requests.
"""
import re
from functools import lru_cache

import numpy as np
import pandas as pd

RAW_COLUMNS = ["transaction_id", "user_id", "transaction_date", "description", "merchant", "amount"]
STATE_CODES = {"CA", "IL", "OH", "GA", "TX", "NY"}
PREFIX = re.compile(r"^(pos purchase|ach debit|card purchase|debit card|visa)\s+")
DAYS = ["Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]


@lru_cache(maxsize=4096)
def clean_description(text):
    """Bank statement text -> merchant words (drops prefixes, store numbers, state codes, symbols)."""
    tokens = str(text).strip().split()
    if len(tokens) > 1 and tokens[-1] in STATE_CODES:
        tokens = tokens[:-1]
    s = " ".join(tokens).lower()
    s = PREFIX.sub("", s)
    s = re.sub(r"#\s*\d+", "", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _group_positions(df, keys):
    """Row positions of each group, in time order (df is already sorted)."""
    return df.groupby(keys, sort=False).indices.values()


def _window(df, keys, window, how="count", inclusive=False):
    """Transactions (count) or amount (sum) in a past time window for each row, current row excluded.
    inclusive=False: window [t - w, t), as in the earlier stages (same-time transactions not counted).
    inclusive=True:  window (t - w, t], counting earlier same-time transactions (refinement features)."""
    t = df["transaction_date"].to_numpy("datetime64[ns]").astype("int64")
    amt = df["amount"].to_numpy(float)
    w = pd.Timedelta(window).value
    out = np.zeros(len(df))
    for pos in _group_positions(df, keys):
        tt = t[pos]
        if inclusive:
            start = np.searchsorted(tt, tt - w, side="right")
            out[pos] = np.arange(len(pos)) - start
        else:
            start = np.searchsorted(tt, tt - w, side="left")
            end = np.searchsorted(tt, tt, side="left")
            if how == "count":
                out[pos] = end - start
            else:
                c = np.concatenate([[0.0], np.cumsum(amt[pos])])
                out[pos] = c[end] - c[start]
    return out


def _prior_mean(df, keys, col):
    """Mean of col over EARLIER rows of the same group (NaN for the first row)."""
    g = df.groupby(keys, sort=False)[col]
    n = g.cumcount().to_numpy(float)
    total = g.cumsum().to_numpy(float) - df[col].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, total / n, np.nan)


def _prior_std(df, keys, col):
    """Sample standard deviation of col over EARLIER rows of the same group (NaN below 2 rows)."""
    x = df[col].to_numpy(float)
    out = np.full(len(df), np.nan)
    for pos in _group_positions(df, keys):
        v = x[pos]
        n = np.arange(len(v), dtype=float)
        s1 = np.concatenate([[0.0], np.cumsum(v)[:-1]])
        s2 = np.concatenate([[0.0], np.cumsum(v * v)[:-1]])
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (s2 - s1 * s1 / n) / (n - 1)
        out[pos] = np.where(n >= 2, np.sqrt(np.clip(var, 0, None)), np.nan)
    return out


def _hour_rarity(hours):
    counts, seen, out = np.zeros(24), 0, []
    for h in hours:
        share = (counts[(h - 1) % 24] + counts[h] + counts[(h + 1) % 24]) / seen if seen else 0.1
        out.append(-np.log(share + 1e-3))
        counts[h] += 1
        seen += 1
    return out


def build_features(df):
    """Compute all features for a chronologically ordered frame of transactions.

    df needs RAW_COLUMNS plus 'category' (true category for past transactions; the
    predicted category, or NaN, for transactions being scored).
    """
    d = df.copy()
    d["transaction_date"] = pd.to_datetime(d["transaction_date"])
    d = d.sort_values(["transaction_date", "transaction_id"]).reset_index(drop=True)
    d["amount"] = d["amount"].astype(float).abs()
    ts = d["transaction_date"]

    # Stage 1 features (time, amount, text)
    d["description_clean"] = d["description"].map(clean_description)
    d["hour"] = ts.dt.hour
    d["day_of_week"] = ts.dt.day_name()
    d["is_weekend"] = d["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    d["is_night"] = (d["hour"] < 6).astype(int)
    d["log_amount"] = np.log1p(d["amount"])
    for day in DAYS:
        d[f"dow_{day}"] = (d["day_of_week"] == day).astype(int)
    d["tx_last_24h"] = _window(d, "user_id", "24h")
    d["is_new_merchant"] = (d.groupby(["user_id", "merchant"]).cumcount() == 0).astype(int)

    # Amount relative to the user's earlier median in the same category (anomaly detection only)
    med = d.groupby(["user_id", "category"], dropna=False)["amount"].transform(
        lambda s: s.shift(1).expanding(min_periods=5).median())
    d["amount_ratio_user_cat"] = (d["amount"] / med).fillna(1.0)
    d["log_amount_ratio"] = np.log(d["amount_ratio_user_cat"])

    # Stage 2 history features
    d["day_of_month"] = ts.dt.day
    d["is_month_start"] = (d["day_of_month"] <= 5).astype(int)
    d["days_since_last_tx"] = (d.groupby("user_id")["transaction_date"].diff().dt.total_seconds() / 86400).fillna(30).clip(upper=30)
    d["days_since_merchant_tx"] = (d.groupby(["user_id", "merchant"])["transaction_date"].diff().dt.total_seconds() / 86400).fillna(90).clip(upper=90)
    d["merchant_visits_prior_log"] = np.log1p(d.groupby(["user_id", "merchant"]).cumcount())
    d["spend_last_24h_log"] = np.log1p(_window(d, "user_id", "24h", "sum"))
    n_prior = d.groupby("user_id").cumcount().to_numpy()
    prior_mean = np.where(n_prior >= 10, _prior_mean(d, "user_id", "log_amount"), np.nan)
    d["log_amount_vs_user"] = (d["log_amount"] - prior_mean).fillna(0)
    prior_hour = d.groupby("user_id")["hour"].transform(lambda s: s.shift(1).expanding(min_periods=10).median())
    d["hour_deviation"] = (d["hour"] - prior_hour).abs().fillna(0)

    # Stage 3 velocity features
    d["tx_last_3h"] = _window(d, "user_id", "3h")
    d["spend_last_3h_log"] = np.log1p(_window(d, "user_id", "3h", "sum"))
    minutes = d.groupby("user_id")["transaction_date"].diff().dt.total_seconds() / 60
    d["minutes_since_last_tx_log"] = np.log1p(minutes.fillna(60 * 24 * 30).clip(upper=60 * 24 * 30))

    # Stage 4 refinement features: merchant profiles (classifier) and behavior features (detector)
    for col in ["log_amount", "hour", "is_weekend", "day_of_month", "is_night", "log_amount_vs_user", "is_month_start"]:
        d[f"merchant_avg_{col}"] = _prior_mean(d, "merchant", col)
    d["merchant_sd_log_amount"] = _prior_std(d, "merchant", "log_amount")
    d["tx_last_1h"] = _window(d, "user_id", "1h", inclusive=True)
    d["tx_last_6h"] = _window(d, "user_id", "6h", inclusive=True)
    d["same_merchant_last_3h"] = _window(d, ["user_id", "merchant"], "3h", inclusive=True)
    d["hour_rarity_user"] = 0.0
    for _, idx in d.groupby("user_id").groups.items():
        d.loc[idx, "hour_rarity_user"] = _hour_rarity(d.loc[idx, "hour"].to_numpy())
    return d
