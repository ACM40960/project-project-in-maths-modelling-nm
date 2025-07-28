import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from adjustText import adjust_text
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve
import lightgbm as lgb

# -------------------------
# Config
# -------------------------
SCORES_PATH = "models/model_scores.csv"
PRED_PATH = "models/final_predictions.csv"
ALL_PRED_PATH = "models/final_predictions_all.csv"
SAVE_DIR = Path("assets/images")
MODEL_PATH = Path("models/booster_0.txt")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load model scores
# -------------------------
df = pd.read_csv(SCORES_PATH)
assert {"Model", "AUROC", "AUPRC"}.issubset(df.columns), "CSV must contain Model, AUROC, and AUPRC columns."

# -------------------------
# 1. AUROC vs AUPRC scatter
# -------------------------
best_model = df.loc[df["AUPRC"].idxmax()]
colors = ['green' if m == best_model["Model"] else 'steelblue' for m in df["Model"]]

plt.figure(figsize=(12, 8))
plt.scatter(df["AUROC"], df["AUPRC"], c=colors, s=120, edgecolor='black')
texts = [plt.text(row["AUROC"], row["AUPRC"], row["Model"], fontsize=10) for _, row in df.iterrows()]
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))
plt.xlabel("AUROC")
plt.ylabel("AUPRC")
plt.title("Model Predictive Performance (AUROC vs AUPRC)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(0.9985, 0.99935)
plt.ylim(df["AUPRC"].min() - 0.001, df["AUPRC"].max() + 0.001)
plt.tight_layout()
plt.savefig(SAVE_DIR / "model_performance_scatter.png")
plt.close()

# -------------------------
# 2. AUPRC bar chart
# -------------------------
plt.figure(figsize=(10, 6))
plt.barh(df["Model"], df["AUPRC"], color='orange', edgecolor='black')
plt.xlabel("AUPRC")
plt.title("Model Evaluation Summary (AUPRC Scores)")
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(SAVE_DIR / "model_auprc_barchart.png")
plt.close()

# -------------------------
# 3. Highlight best AUPRC
# -------------------------
colors = ['green' if model == best_model["Model"] else 'gray' for model in df["Model"]]
plt.figure(figsize=(10, 6))
plt.barh(df["Model"], df["AUPRC"], color=colors, edgecolor='black')
plt.xlabel("AUPRC")
plt.title("Best-Performing Model Highlighted by AUPRC")
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(SAVE_DIR / "model_best_auprc_highlighted.png")
plt.close()

# -------------------------
# 4–7: Real Evaluation Curves for best model
# -------------------------
pred_df = pd.read_csv(PRED_PATH)
y_true = pred_df["y_true"].values
y_prob = pred_df["y_prob"].values

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkblue", label=f"AUROC = {roc_auc_score(y_true, y_prob):.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve – Best Model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_DIR / "roc_curve.png")
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color="teal", label=f"AUPRC = {average_precision_score(y_true, y_prob):.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision vs Recall – Best Model")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR / "pr_curve.png")
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_true, y_prob > 0.5)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
plt.title("Confusion Matrix on Test Data")
plt.tight_layout()
plt.savefig(SAVE_DIR / "confusion_matrix.png")
plt.close()

# Calibration Curve
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', color='firebrick', label="Model")
plt.plot([0, 1], [0, 1], linestyle='--', label="Ideal")
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Model Calibration Curve – Ideal vs Predicted Probabilities")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_DIR / "calibration_curve.png")
plt.close()

# -------------------------
# 8. LightGBM Feature Importance (Gain)
# -------------------------
try:
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    ax = lgb.plot_importance(booster, importance_type="gain", max_num_features=10, figsize=(10, 7), color="teal")
    ax.set_title("Top Features Based on LightGBM Gain Importance", fontsize=14)
    ax.figure.tight_layout()
    ax.figure.savefig(SAVE_DIR / "lgbm_feature_importance.png")
    plt.close()
except Exception as e:
    print("⚠️ Could not plot feature importance:", e)

# -------------------------
# 9. All Model ROC + PR Curves
# -------------------------
try:
    all_preds = pd.read_csv(ALL_PRED_PATH)
    all_curves = all_preds.groupby("Model")

    # ROC
    plt.figure(figsize=(10, 8))
    for name, group in all_curves:
        fpr, tpr, _ = roc_curve(group["y_true"], group["y_prob"])
        auroc = roc_auc_score(group["y_true"], group["y_prob"])
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUROC={auroc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for All Models")
    plt.xlim(0.0, 0.1)
    plt.ylim(0.9, 1.01)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "roc_curves_all_models.png")
    plt.close()

    # PR
    plt.figure(figsize=(10, 8))
    for name, group in all_curves:
        precision, recall, _ = precision_recall_curve(group["y_true"], group["y_prob"])
        auprc = average_precision_score(group["y_true"], group["y_prob"])
        plt.plot(recall, precision, lw=2, label=f"{name} (AUPRC={auprc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves for All Models")
    plt.xlim(0.0, 1)
    plt.ylim(0.9, 1.01)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "pr_curves_all_models.png")
    plt.close()

except Exception as e:
    print("⚠️ Could not plot ROC/PR curves for all models:", e)

print(f"✅ Plots saved to {SAVE_DIR.resolve()}")
