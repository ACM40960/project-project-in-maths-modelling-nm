"""
Feature-engineering functions for BankSim
----------------------------------------
Produces temporal + lightweight relational features identical to the
notebook version used during model development.
"""
import pandas as pd, numpy as np
from typing import List

# ------------------------------------------------------------------
# main "engineer" function
# ------------------------------------------------------------------
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- basic cleanup expected on fresh CSV ----------------------
    obj_cols = ["customer", "merchant", "age", "gender", "category"]
    for c in obj_cols + ["zipcodeOri", "zipMerchant"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip("'")

    df["zipcodeOri"]  = df["zipcodeOri"].astype(int)
    df["zipMerchant"] = df["zipMerchant"].astype(int)

    for col in ["customer", "merchant"]:
        df[col] = df[col].astype("category").cat.codes

    df = pd.get_dummies(df, columns=["age", "gender", "category"], drop_first=True)

    # ---- temporal & relational -----------------------------------
    df = df.sort_values(["customer", "step"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(df["step"], unit="h")

    df["time_since_last_txn"] = (
        df.groupby("customer")["timestamp"].diff().dt.total_seconds().fillna(0)
    )
    df["is_night_txn"] = df["timestamp"].dt.hour.between(0, 5).astype(int)

    # 1-hour rolling count
    df["txn_count_1h"] = (
        df.groupby("customer", group_keys=False)
        .apply(lambda g: g.rolling("1h", on="timestamp").amount.count())
        .reset_index(drop=True)
        .astype(int)
    )

    # 24-hour rolling sum
    df["txn_amount_sum_24h"] = (
        df.groupby("customer", group_keys=False)
        .apply(lambda g: g.rolling("24h", on="timestamp").amount.sum())
        .reset_index(drop=True)
    )

    # 24-hour rolling Z-score
    def z24(g):
        roll = g.rolling("24h", on="timestamp").amount
        mu, sd = roll.mean(), roll.std()
        return np.where(sd > 0, (g["amount"] - mu) / sd, 0.0)

    df["amount_zscore_24h"] = (
        df.groupby("customer", group_keys=False).apply(z24).reset_index(drop=True)
    )

    # relational counts
    df["pair_txn_count_sofar"] = df.groupby(["customer", "merchant"]).cumcount()
    df["is_first_interaction"] = (df["pair_txn_count_sofar"] == 0).astype(int)
    df["cust_txn_count_sofar"]  = df.groupby("customer").cumcount()  + 1
    df["merch_txn_count_sofar"] = df.groupby("merchant").cumcount() + 1

    # final numeric z-score type fix
    df["amount_zscore_24h"] = pd.to_numeric(df["amount_zscore_24h"], errors="coerce").fillna(0.0)

    # drop helper
    df = df.drop(columns=["timestamp"])

    return df

# list of numeric/boolean columns after engineering (filled at runtime)
def feature_list(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("fraud",)]
