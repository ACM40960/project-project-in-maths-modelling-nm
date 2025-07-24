"""
Feature engineering for BankSim fraud detector.
Adds temporal and lightweight relational columns.
"""

import pandas as pd
import numpy as np
from typing import List

# ------------------------------------------------------------------
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---------- basic clean ---------------------------------------
    obj_cols = ["customer", "merchant", "age", "gender", "category"]
    for col in obj_cols + ["zipcodeOri", "zipMerchant"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip("'")

    df["zipcodeOri"]  = df["zipcodeOri"].astype(int)
    df["zipMerchant"] = df["zipMerchant"].astype(int)

    for col in ["customer", "merchant"]:
        df[col] = df[col].astype("category").cat.codes

    df = pd.get_dummies(df, columns=["age", "gender", "category"], drop_first=True)

    # ---------- temporal features ---------------------------------
    df = df.sort_values(["customer", "step"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(df["step"], unit="h")

    df["time_since_last_txn"] = (
        df.groupby("customer")["timestamp"].diff().dt.total_seconds().fillna(0)
    )
    df["is_night_txn"] = df["timestamp"].dt.hour.between(0, 5).astype(int)

    df["txn_count_1h"] = (
        df.groupby("customer", group_keys=False, include_groups=True)
        .apply(lambda g: g.rolling("1h", on="timestamp").amount.count())
        .reset_index(drop=True)
        .astype(int)
    )

    df["txn_amount_sum_24h"] = (
        df.groupby("customer", group_keys=False, include_groups=True)
        .apply(lambda g: g.rolling("24h", on="timestamp").amount.sum())
        .reset_index(drop=True)
    )

    def z24(g):
        roll = g.rolling("24h", on="timestamp").amount
        mu, sd = roll.mean(), roll.std()
        return np.where(sd > 0, (g["amount"] - mu) / sd, 0.0)

    df["amount_zscore_24h"] = (
        df.groupby("customer", group_keys=False, include_groups=True)
        .apply(z24)
        .reset_index(drop=True)
    )

    # ---------- relational features -------------------------------
    df["pair_txn_count_sofar"] = df.groupby(["customer", "merchant"]).cumcount()
    df["is_first_interaction"] = (df["pair_txn_count_sofar"] == 0).astype(int)
    df["cust_txn_count_sofar"]  = df.groupby("customer").cumcount()  + 1
    df["merch_txn_count_sofar"] = df.groupby("merchant").cumcount() + 1

    # ensure numeric dtype
    df["amount_zscore_24h"] = pd.to_numeric(df["amount_zscore_24h"], errors="coerce").fillna(0.0)

    # drop helper timestamp
    df = df.drop(columns=["timestamp"])

    return df


# helper to get the ordered feature list after engineering
def feature_list(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != "fraud"]
