import json, numpy as np
from pathlib import Path
import lightgbm as lgb
from typing import List
from .config import MODEL_DIR, DEFAULT_THRESHOLD, NUM_MODELS, UNDERSAMPLE_RATIO
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_recall_curve
from .features import feature_list

# ----------------------------------------------------------------
def train_ensemble(X, y, ratio=UNDERSAMPLE_RATIO, n_models=NUM_MODELS) -> List[lgb.Booster]:
    boosters = []
    for k in range(n_models):
        rus = RandomUnderSampler(sampling_strategy=ratio, random_state=200+k)
        X_r, y_r = rus.fit_resample(X, y)
        lgbm = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.07,
            num_leaves=63, random_state=200+k, verbose=-1)
        lgbm.fit(X_r, y_r)
        boosters.append(lgbm.booster_)
    return boosters

# ----------------------------------------------------------------
def choose_threshold(boosters, X_val, y_val, min_precision=0.90):
    proba = np.mean([b.predict(X_val) for b in boosters], axis=0)
    prec, rec, thr = precision_recall_curve(y_val, proba)
    idx  = np.where(prec >= min_precision)[0]
    best = idx[np.argmax(rec[idx])]        # highest recall among those points
    return float(thr[best])


# ----------------------------------------------------------------
def save(boosters: List[lgb.Booster], threshold: float):
    MODEL_DIR.mkdir(exist_ok=True)
    for i, b in enumerate(boosters):
        b.save_model(MODEL_DIR / f"booster_{i}.txt")
    (MODEL_DIR / "threshold.json").write_text(json.dumps({"threshold": threshold}))

# ----------------------------------------------------------------
def load():
    boosters = [lgb.Booster(model_file=str(p))
                for p in sorted(MODEL_DIR.glob("booster_*.txt"))]
    if (MODEL_DIR / "threshold.json").exists():
        thr = json.loads((MODEL_DIR / "threshold.json").read_text())["threshold"]
    else:
        thr = DEFAULT_THRESHOLD
    return boosters, thr

# ----------------------------------------------------------------
def predict_proba(boosters: List[lgb.Booster], row_vec: np.ndarray) -> float:
    return float(np.mean([b.predict(row_vec.reshape(1, -1)) for b in boosters]))

def classify(boosters: List[lgb.Booster], threshold: float, row_vec: np.ndarray):
    p = predict_proba(boosters, row_vec)
    return int(p >= threshold), p

# ----------------------------------------------------------------
def predict_batch(boosters: List[lgb.Booster], X: np.ndarray) -> np.ndarray:
    preds = np.stack([b.predict(X) for b in boosters])
    return np.mean(preds, axis=0)

