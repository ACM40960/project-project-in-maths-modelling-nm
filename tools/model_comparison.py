"""
model_comparison.py

This script compares four fraud detection modeling strategies on the preprocessed BankSim dataset:
    1. Weighted LightGBM (Optuna)
    2. Ensemble of undersampled LightGBMs
    3. SMOTE + LightGBM
    4. Weighted XGBoost (Optuna)

Outputs a sorted score table based on AUPRC and saves results as CSV.
Also saves per-model predictions for evaluation curve plotting.
"""

import sys
import pandas as pd
import numpy as np
import optuna
import warnings
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

# Add src to path for local imports
sys.path.append("src")
from banksim_fraud.features import engineer, feature_list

# Define metrics
def pr_auc(y_true, y_proba):
    return average_precision_score(y_true, y_proba)

# Load raw data and engineer features
RAW_PATH = "data/bs140513_032310.csv"
SAVE_PATH = "models/model_scores.csv"
PRED_PATH = "models/final_predictions.csv"
PRED_ALL_PATH = "models/final_predictions_all.csv"

data_raw = pd.read_csv(RAW_PATH)
data = engineer(data_raw)
num_cols = feature_list(data)

# Out-of-time split
step_cut = data["step"].quantile(0.80)
X_train = data[data["step"] <= step_cut][num_cols]
y_train = data[data["step"] <= step_cut]["fraud"].astype(int)
X_test  = data[data["step"] > step_cut][num_cols]
y_test  = data[data["step"] > step_cut]["fraud"].astype(int)

assert len(X_test) > 0, "Test set is empty."

# Class imbalance ratio
pos_w = (y_train == 0).sum() / (y_train == 1).sum()
scores = []
predictions = []  # collect model name + predictions

# -------------------------------------------------------------------
# 1. Weighted LightGBM (Optuna)
# -------------------------------------------------------------------
def obj_lgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estim", 300, 600, 100),
        "num_leaves": trial.suggest_int("leaves", 31, 127, 32),
        "learning_rate": trial.suggest_float("lr", 0.03, 0.1, log=True),
        "subsample": trial.suggest_float("sub", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("col", 0.6, 1.0),
        "scale_pos_weight": pos_w,
        "objective": "binary",
        "random_state": 42,
        "verbose": -1,
    }
    mdl = LGBMClassifier(**params)
    mdl.fit(X_train, y_train)
    return pr_auc(y_test, mdl.predict_proba(X_test)[:, 1])

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(obj_lgb, n_trials=20)

lgb_best = LGBMClassifier(**study.best_params, scale_pos_weight=pos_w, objective="binary", random_state=42)
lgb_best.fit(X_train, y_train)
pred_lgb = lgb_best.predict_proba(X_test)[:, 1]
scores.append(("LGB-Weighted", pr_auc(y_test, pred_lgb), roc_auc_score(y_test, pred_lgb)))
predictions.append(("LGB-Weighted", pred_lgb))

# -------------------------------------------------------------------
# 2. Undersample Ensemble (5× LGBM)
# -------------------------------------------------------------------
def lgb_ensemble(ratio):
    out = np.zeros(len(y_test))
    for k in range(5):
        rus = RandomUnderSampler(sampling_strategy=ratio, random_state=100 + k)
        X_r, y_r = rus.fit_resample(X_train, y_train)
        mdl = LGBMClassifier(n_estimators=300, learning_rate=0.07, num_leaves=63, random_state=100 + k)
        mdl.fit(X_r, y_r)
        out += mdl.predict_proba(X_test)[:, 1]
    return out / 5

for ratio in [0.02, 0.05, 0.10]:
    pred_eu = lgb_ensemble(ratio)
    model_name = f"LGB-EU {ratio}"
    scores.append((model_name, pr_auc(y_test, pred_eu), roc_auc_score(y_test, pred_eu)))
    predictions.append((model_name, pred_eu))

# -------------------------------------------------------------------
# 3. SMOTE + LightGBM
# -------------------------------------------------------------------
sm = SMOTE(sampling_strategy=0.10, random_state=42)
X_sm, y_sm = sm.fit_resample(X_train, y_train)

lgb_sm = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, random_state=42)
lgb_sm.fit(X_sm, y_sm)
pred_sm = lgb_sm.predict_proba(X_test)[:, 1]
scores.append(("LGB-SMOTE 0.10", pr_auc(y_test, pred_sm), roc_auc_score(y_test, pred_sm)))
predictions.append(("LGB-SMOTE 0.10", pred_sm))

# -------------------------------------------------------------------
# 4. Weighted XGBoost (Optuna)
# -------------------------------------------------------------------
def obj_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estim", 300, 600, 100),
        "max_depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("lr", 0.04, 0.2, log=True),
        "subsample": trial.suggest_float("sub", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("col", 0.6, 1.0),
        "scale_pos_weight": pos_w,
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
    }
    mdl = XGBClassifier(**params)
    mdl.fit(X_train, y_train)
    return pr_auc(y_test, mdl.predict_proba(X_test)[:, 1])

study2 = optuna.create_study(direction="maximize")
study2.optimize(obj_xgb, n_trials=20)

xgb_best = XGBClassifier(**study2.best_params, scale_pos_weight=pos_w, eval_metric="aucpr", random_state=42, n_jobs=-1)
xgb_best.fit(X_train, y_train)
pred_xgb = xgb_best.predict_proba(X_test)[:, 1]
scores.append(("XGB-Weighted", pr_auc(y_test, pred_xgb), roc_auc_score(y_test, pred_xgb)))
predictions.append(("XGB-Weighted", pred_xgb))

# -------------------------------------------------------------------
# Save + Print Score Table
# -------------------------------------------------------------------
results = pd.DataFrame(scores, columns=["Model", "AUPRC", "AUROC"])
results.to_csv(SAVE_PATH, index=False)

print("\n=== Out-of-Time Model Comparison ===")
print(results.sort_values("AUPRC", ascending=False).to_string(index=False))

# -------------------------------------------------------------------
# Save per-model predictions (for plotting PR/ROC curves)
# -------------------------------------------------------------------
all_rows = []
for name, preds in predictions:
    for yt, yp in zip(y_test, preds):
        all_rows.append({"Model": name, "y_true": yt, "y_prob": yp})

pd.DataFrame(all_rows).to_csv(PRED_ALL_PATH, index=False)
print(f"✅ Saved: {PRED_ALL_PATH}")

# -------------------------------------------------------------------
# Save best model predictions (standalone file for app or baseline eval)
# -------------------------------------------------------------------
best_idx = results["AUPRC"].idxmax()
best_model_name = results.loc[best_idx, "Model"]
best_pred = dict(predictions)[best_model_name]

print(f"\nSaving predictions from best model: {best_model_name}")
pd.DataFrame({
    "y_true": y_test,
    "y_prob": best_pred
}).to_csv(PRED_PATH, index=False)
print(f"✅ Saved: {PRED_PATH}")
