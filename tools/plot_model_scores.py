import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve
from matplotlib.gridspec import GridSpec

# -------------------------
# Config
# -------------------------
SCORES_PATH = "models/model_scores.csv"
PRED_PATH = "models/final_predictions.csv"
ALL_PRED_PATH = "models/final_predictions_all.csv"
SAVE_DIR = Path("assets/images")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv(SCORES_PATH)
pred_df = pd.read_csv(PRED_PATH)
y_true = pred_df["y_true"].values
y_prob = pred_df["y_prob"].values
best_model = df.loc[df["AUPRC"].idxmax()]

# -------------------------
# Recalculate Curves
# -------------------------
fpr, tpr, _ = roc_curve(y_true, y_prob)
precision, recall, _ = precision_recall_curve(y_true, y_prob)
cm = confusion_matrix(y_true, y_prob > 0.5)
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

# -------------------------
# 1. Combined ROC + PR Curve (Best Model)
# -------------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].plot(fpr, tpr, color="darkblue", label=f"AUROC = {roc_auc_score(y_true, y_prob):.4f}")
axs[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axs[0].set_xlabel("False Positive Rate")
axs[0].set_ylabel("True Positive Rate")
axs[0].set_title("ROC Curve")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(recall, precision, color="teal", label=f"AUPRC = {average_precision_score(y_true, y_prob):.4f}")
axs[1].set_xlabel("Recall")
axs[1].set_ylabel("Precision")
axs[1].set_title("Precision-Recall Curve")
axs[1].legend()
axs[1].grid(True)

fig.suptitle("Best Model Evaluation: ROC & PR Curves", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(SAVE_DIR / "combined_roc_pr_curve.png")
plt.close()

# -------------------------
# 2. Calibration + Confusion Matrix Combined
# -------------------------
fig = plt.figure(figsize=(13, 6))
gs = GridSpec(1, 2, width_ratios=[1.2, 1])

ax1 = fig.add_subplot(gs[0])
ax1.plot(prob_pred, prob_true, marker='o', color='firebrick', label="Model")
ax1.plot([0, 1], [0, 1], linestyle='--', label="Ideal")
ax1.set_xlabel("Predicted Probability")
ax1.set_ylabel("True Probability")
ax1.set_title("Calibration Curve")
ax1.legend()
ax1.grid(True)

ax2 = fig.add_subplot(gs[1])
ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax2)
ax2.set_title("Confusion Matrix")

fig.suptitle("Model Reliability & Classification Summary", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(SAVE_DIR / "combined_calibration_confusion.png")
plt.close()

# -------------------------
# 3. AUROC vs AUPRC Scatter + Highlighted AUPRC Bar Chart
# -------------------------
fig, axs = plt.subplots(1, 2, figsize=(16, 7))
colors = ['green' if m == best_model["Model"] else 'steelblue' for m in df["Model"]]

# AUROC vs AUPRC Scatter
axs[0].scatter(df["AUROC"], df["AUPRC"], c=colors, s=100, edgecolor='black')
for _, row in df.iterrows():
    axs[0].text(row["AUROC"], row["AUPRC"], row["Model"], fontsize=9)
axs[0].set_xlabel("AUROC")
axs[0].set_ylabel("AUPRC")
axs[0].set_title("Model Predictive Performance")
axs[0].grid(True, linestyle='--', alpha=0.5)
axs[0].set_xlim(0.9985, 0.99935)
axs[0].set_ylim(df["AUPRC"].min() - 0.001, df["AUPRC"].max() + 0.001)

# AUPRC Bar Chart with Highlight
highlight_colors = ['green' if m == best_model["Model"] else 'gray' for m in df["Model"]]
axs[1].barh(df["Model"], df["AUPRC"], color=highlight_colors, edgecolor='black')
axs[1].set_xlabel("AUPRC")
axs[1].set_title("AUPRC Scores (Best Model Highlighted)")
axs[1].grid(True, axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(SAVE_DIR / "combined_auroc_auprc_summary.png")
plt.close()

# -------------------------
# 4. All Models ROC + PR Curves
# -------------------------
try:
    all_preds = pd.read_csv(ALL_PRED_PATH)
    all_curves = all_preds.groupby("Model")

    # ROC Curves
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

    # PR Curves
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

# -------------------------
# 5. LightGBM Feature Importance (Gain)
# -------------------------
import lightgbm as lgb

MODEL_PATH = Path("models/booster_0.txt")

try:
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    ax = lgb.plot_importance(
        booster,
        importance_type="gain",
        max_num_features=10,
        figsize=(10, 7),
        color="teal"
    )
    ax.set_title("Top Features Based on LightGBM Gain Importance", fontsize=14)
    ax.figure.tight_layout()
    ax.figure.savefig(SAVE_DIR / "lgbm_feature_importance.png")
    plt.close()
except Exception as e:
    print("⚠️ Could not plot feature importance:", e)


print(f"✅ Combined plots saved to: {SAVE_DIR.resolve()}")
