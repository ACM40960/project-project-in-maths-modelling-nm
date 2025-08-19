"""
model_comparison.py

Compare and tune fraud detection strategies on BankSim:

    1) Weighted LightGBM  (Optuna)
    2) EU-LightGBM ensemble (Optuna over ratio + LGBM params)
    3) SMOTE + LightGBM   (Optuna over sampling ratio + LGBM params)
    4) Weighted XGBoost   (Optuna)

All objectives tune on a time-aware validation split from TRAIN only,
then we retrain on full TRAIN and score on OOT TEST.

Outputs:
- models/model_scores.csv
- models/final_predictions.csv           (best model)
- models/final_predictions_all.csv       (all models’ probs for curve plotting)
"""

import sys
import warnings
import numpy as np
import pandas as pd
import optuna

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

# Local imports
sys.path.append("src")
from banksim_fraud.features import engineer, feature_list

# ---------------------------
# Config / IO
# ---------------------------
RAW_PATH = "data/bs140513_032310.csv"
SAVE_PATH = "models/model_scores.csv"
PRED_PATH = "models/final_predictions.csv"
PRED_ALL_PATH = "models/final_predictions_all.csv"

# ---------------------------
# Helpers
# ---------------------------
def pr_auc(y_true, y_proba):
    return average_precision_score(y_true, y_proba)

def time_aware_split(df, feat_cols, label_col="fraud", test_quantile=0.80, val_quantile=0.70):
    """
    Split into: train_inner (<= val_cut), val_inner (> val_cut, <= test_cut), test_outer (> test_cut)
    All splits are by 'step' to preserve time order.
    """
    test_cut = df["step"].quantile(test_quantile)
    val_cut = df["step"].quantile(val_quantile)

    train_mask = df["step"] <= val_cut
    val_mask   = (df["step"] > val_cut) & (df["step"] <= test_cut)
    test_mask  = df["step"] > test_cut

    X_tr = df.loc[train_mask, feat_cols]
    y_tr = df.loc[train_mask, label_col].astype(int)
    X_va = df.loc[val_mask, feat_cols]
    y_va = df.loc[val_mask, label_col].astype(int)
    X_te = df.loc[test_mask, feat_cols]
    y_te = df.loc[test_mask, label_col].astype(int)

    return (X_tr, y_tr, X_va, y_va, X_te, y_te, val_cut, test_cut)

# ---------------------------
# Load & FE
# ---------------------------
data_raw = pd.read_csv(RAW_PATH)
data = engineer(data_raw)
num_cols = feature_list(data)

# Build time-aware splits
X_tr, y_tr, X_val, y_val, X_test, y_test, val_cut, test_cut = time_aware_split(
    data, feat_cols=num_cols, label_col="fraud", test_quantile=0.80, val_quantile=0.70
)
assert len(X_test) > 0, "Test set is empty."

# Class imbalance ratio for weighted objectives
pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

# For retraining on full TRAIN (train+val) after tuning
trainval_mask = data["step"] <= test_cut
X_trainval = data.loc[trainval_mask, num_cols]
y_trainval = data.loc[trainval_mask, "fraud"].astype(int)

# Storage
scores = []
predictions = []   # (name, y_proba_test)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# 1) Weighted LightGBM (Optuna)
# ============================================================
def obj_lgb_weighted(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.95),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 300),
        "objective": "binary",
        "scale_pos_weight": pos_w,
        "verbose": -1,
    }
    mdl = LGBMClassifier(**params, random_state=42)
    mdl.fit(X_tr, y_tr)
    val_pred = mdl.predict_proba(X_val)[:, 1]
    return pr_auc(y_val, val_pred)

study_lgb_w = optuna.create_study(direction="maximize")
study_lgb_w.optimize(obj_lgb_weighted, n_trials=30)

best_lgb_w = LGBMClassifier(**study_lgb_w.best_params, random_state=42, scale_pos_weight=pos_w, objective="binary")
best_lgb_w.fit(X_trainval, y_trainval)
pred_lgb_w = best_lgb_w.predict_proba(X_test)[:, 1]
scores.append(("LGB-Weighted", pr_auc(y_test, pred_lgb_w), roc_auc_score(y_test, pred_lgb_w)))
predictions.append(("LGB-Weighted", pred_lgb_w))

# ============================================================
# 2) EU-LightGBM Ensemble (Optuna)
# ============================================================
def obj_lgb_eu(trial):
    ratio = trial.suggest_categorical("ratio", [0.02, 0.05, 0.10])
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.95),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 300),
        "objective": "binary",
        "verbose": -1,
    }

    val_preds = np.zeros(len(y_val), dtype=float)
    for k in range(5):
        rus = RandomUnderSampler(sampling_strategy=ratio, random_state=100 + k)
        X_r, y_r = rus.fit_resample(X_tr, y_tr)
        mdl = LGBMClassifier(**params, random_state=100 + k)
        mdl.fit(X_r, y_r)
        val_preds += mdl.predict_proba(X_val)[:, 1]
    val_preds /= 5.0
    return pr_auc(y_val, val_preds)

study_lgb_eu = optuna.create_study(direction="maximize")
study_lgb_eu.optimize(obj_lgb_eu, n_trials=30)

best_ratio = study_lgb_eu.best_params["ratio"]
eu_params = {k: v for k, v in study_lgb_eu.best_params.items() if k != "ratio"}

test_preds_eu = np.zeros(len(y_test), dtype=float)
for k in range(5):
    rus = RandomUnderSampler(sampling_strategy=best_ratio, random_state=200 + k)
    X_rv, y_rv = rus.fit_resample(X_trainval, y_trainval)
    mdl = LGBMClassifier(**eu_params, random_state=200 + k)
    mdl.fit(X_rv, y_rv)
    test_preds_eu += mdl.predict_proba(X_test)[:, 1]
test_preds_eu /= 5.0

scores.append((f"LGB-EU {best_ratio}", pr_auc(y_test, test_preds_eu), roc_auc_score(y_test, test_preds_eu)))
predictions.append((f"LGB-EU {best_ratio}", test_preds_eu))

# ============================================================
# 3) SMOTE + LightGBM (Optuna)
# ============================================================
def obj_lgb_smote(trial):
    smote_ratio = trial.suggest_float("smote_ratio", 0.05, 0.20, step=0.05)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.95),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 300),
        "objective": "binary",
        "verbose": -1,
    }
    sm = SMOTE(sampling_strategy=smote_ratio, random_state=42)
    X_sm, y_sm = sm.fit_resample(X_tr, y_tr)
    mdl = LGBMClassifier(**params, random_state=42)
    mdl.fit(X_sm, y_sm)
    val_pred = mdl.predict_proba(X_val)[:, 1]
    return pr_auc(y_val, val_pred)

study_lgb_sm = optuna.create_study(direction="maximize")
study_lgb_sm.optimize(obj_lgb_smote, n_trials=30)

best_sm_ratio = study_lgb_sm.best_params["smote_ratio"]
sm_params = {k: v for k, v in study_lgb_sm.best_params.items() if k != "smote_ratio"}

sm = SMOTE(sampling_strategy=best_sm_ratio, random_state=42)
X_sm_trainval, y_sm_trainval = sm.fit_resample(X_trainval, y_trainval)
best_lgb_sm = LGBMClassifier(**sm_params, random_state=42)
best_lgb_sm.fit(X_sm_trainval, y_sm_trainval)
pred_lgb_sm = best_lgb_sm.predict_proba(X_test)[:, 1]
scores.append((f"LGB-SMOTE {best_sm_ratio:.2f}", pr_auc(y_test, pred_lgb_sm), roc_auc_score(y_test, pred_lgb_sm)))
predictions.append((f"LGB-SMOTE {best_sm_ratio:.2f}", pred_lgb_sm))

# ============================================================
# 4) Weighted XGBoost (Optuna)
# ============================================================
def obj_xgb_weighted(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "scale_pos_weight": pos_w,
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    mdl = XGBClassifier(**params)
    mdl.fit(X_tr, y_tr)
    val_pred = mdl.predict_proba(X_val)[:, 1]
    return pr_auc(y_val, val_pred)

study_xgb_w = optuna.create_study(direction="maximize")
study_xgb_w.optimize(obj_xgb_weighted, n_trials=30)

best_xgb_w = XGBClassifier(**study_xgb_w.best_params)
best_xgb_w.fit(X_trainval, y_trainval)
pred_xgb_w = best_xgb_w.predict_proba(X_test)[:, 1]
scores.append(("XGB-Weighted", pr_auc(y_test, pred_xgb_w), roc_auc_score(y_test, pred_xgb_w)))
predictions.append(("XGB-Weighted", pred_xgb_w))

# ============================================================
# Save results
# ============================================================
results = pd.DataFrame(scores, columns=["Model", "AUPRC", "AUROC"])
results_sorted = results.sort_values("AUPRC", ascending=False)
results_sorted.to_csv(SAVE_PATH, index=False)

print("\n=== Out-of-Time Model Comparison (TEST > {:.0f}th pct of step) ===".format(100 * 0.80))
print(results_sorted.to_string(index=False))

# Save per-model predictions
rows = []
for name, preds in predictions:
    for yt, yp in zip(y_test, preds):
        rows.append({"Model": name, "y_true": int(yt), "y_prob": float(yp)})
pd.DataFrame(rows).to_csv(PRED_ALL_PATH, index=False)
print(f"\nSaved per-model probabilities: {PRED_ALL_PATH}")

# Save best model predictions
best_name = results_sorted.iloc[0]["Model"]
best_pred = dict(predictions)[best_name]
pd.DataFrame({"y_true": y_test.values, "y_prob": best_pred}).to_csv(PRED_PATH, index=False)
print(f"Best model: {best_name}")
print(f"Saved best predictions: {PRED_PATH}")
