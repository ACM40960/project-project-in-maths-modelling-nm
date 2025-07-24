"""
FastAPI micro-service for real-time BankSim fraud scoring
--------------------------------------------------------
• POST /score  – returns {"fraud_flag": bool, "probability": float}
• GET  /alive  – health-check

The request payload contains only the 15 raw BankSim fields.
This service:
  1. Re-creates **all 24 engineered features** (temporal + relational).
  2. Pads missing one-hot columns so every vector has the **39 columns**
     used in training (loaded from models/features.json).
  3. Runs the 5-model LightGBM ensemble and applies the locked threshold.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Union, Deque, Tuple, Dict
from collections import defaultdict, deque
from datetime import datetime, timedelta
import hashlib, json, numpy as np, pandas as pd
import pathlib

from .model import load, classify
from .features import engineer  # reuse notebook feature logic

# ------------------------------------------------------------------
# Load model & threshold once
boosters, THRESHOLD = load()

# Load the master feature list saved during training
MODEL_DIR = pathlib.Path(__file__).resolve().parent / "models"
with open(MODEL_DIR / "features.json") as f:
    FEATURES = json.load(f)                  # ordered 39-column schema

# ------------------------------------------------------------------
app = FastAPI(title="BankSim Fraud API", version="0.1.0")

# Rolling state for temporal / relational features
_last_ts: Dict[int, datetime]                       = {}
_cust_buf: Dict[int, Deque[Tuple[datetime, float]]] = defaultdict(lambda: deque(maxlen=500))
_pair_cnt: Dict[Tuple[int, int], int]               = defaultdict(int)
_cust_cnt: Dict[int, int]                           = defaultdict(int)
_merch_cnt: Dict[int, int]                          = defaultdict(int)

EPOCH = datetime(2025, 1, 1)  # same anchor day used in training

# ------------------------------------------------------------------
# Helper: stable hash → 32-bit int
def _hash_id(x) -> int:
    if isinstance(x, int):
        return x
    h = hashlib.md5(str(x).encode()).hexdigest()  # 128-bit hex
    return int(h[:8], 16)                         # first 32 bits as int

# ------------------------------------------------------------------
# Pydantic schemas
class Txn(BaseModel):
    step:        int
    customer:    Union[int, str]
    merchant:    Union[int, str]
    age:         str
    gender:      str
    category:    str
    amount:      float
    zipcodeOri:  int = Field(alias="zipcodeOri")
    zipMerchant: int = Field(alias="zipMerchant")
    fraud:       Union[int, None] = None  # optional, ignored

class ScoreOut(BaseModel):
    fraud_flag:  bool
    probability: float

# ------------------------------------------------------------------
@app.get("/alive")
def alive():
    return {"status": "ok"}

# ------------------------------------------------------------------
def update_state_and_features(tx: Txn) -> pd.Series:
    """Return one row with *all* engineered features (39 cols)."""
    cust  = _hash_id(tx.customer)
    merch = _hash_id(tx.merchant)
    step  = tx.step
    ts    = EPOCH + timedelta(hours=step)

    # rolling buffer per customer (24 h window)
    buf = _cust_buf[cust]
    buf.append((ts, tx.amount))
    while buf and buf[0][0] < ts - timedelta(hours=24):
        buf.popleft()

    # 1-hour metrics
    cnt_1h  = sum(1 for t, _ in buf if t >= ts - timedelta(hours=1))
    sum_24h = sum(a for _, a in buf)

    # z-score
    amounts = [a for _, a in buf]
    mu, sd  = (np.mean(amounts), np.std(amounts)) if amounts else (0.0, 1.0)
    z24     = (tx.amount - mu) / sd if sd > 0 else 0.0

    # time since last txn
    tslast = (ts - _last_ts.get(cust, ts)).total_seconds() if cust in _last_ts else 0.0
    _last_ts[cust] = ts

    # relational counts
    pair_key = (cust, merch)
    pair_cnt = _pair_cnt[pair_key]
    _pair_cnt[pair_key] += 1
    _cust_cnt[cust]  += 1
    _merch_cnt[merch] += 1

    row = {
        # raw columns
        "step": step,
        "customer": cust,
        "merchant": merch,
        "amount": tx.amount,
        "zipcodeOri": tx.zipcodeOri,
        "zipMerchant": tx.zipMerchant,
        # engineered
        "time_since_last_txn": tslast,
        "is_night_txn": int(ts.hour < 6),
        "txn_count_1h": cnt_1h,
        "txn_amount_sum_24h": sum_24h,
        "amount_zscore_24h": z24,
        "pair_txn_count_sofar": pair_cnt,
        "is_first_interaction": int(pair_cnt == 0),
        "cust_txn_count_sofar": _cust_cnt[cust],
        "merch_txn_count_sofar": _merch_cnt[merch],
        # one-hots (initialised to 0, set to 1 where applicable)
    }
    row[f"age_{tx.age}"]        = 1
    row[f"gender_{tx.gender}"]  = 1
    row[f"category_{tx.category}"] = 1

    return pd.Series(row)

# ------------------------------------------------------------------
@app.post("/score", response_model=ScoreOut)
def score(txn: Txn):
    # 1. engineer full feature row
    row = update_state_and_features(txn)

    # 2. pad & order
    for col in FEATURES:
        if col not in row:
            row[col] = 0.0
    vec = row[FEATURES].to_numpy(dtype=float)

    # 3. classify
    label, prob = classify(boosters, THRESHOLD, vec)
    return {"fraud_flag": bool(label), "probability": round(prob, 6)}
