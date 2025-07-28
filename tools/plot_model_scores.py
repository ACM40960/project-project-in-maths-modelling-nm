"""
plot_model_scores.py

Reads model performance metrics from CSV and generates 3 visualizations:
1. AUROC vs AUPRC scatter plot
2. AUPRC bar chart (all models)
3. AUPRC bar chart with best model highlighted

Figures are saved to: assets/images/
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# Config
# -------------------------
CSV_PATH = "data/all_model_scores.csv"
SAVE_DIR = Path("assets/images")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load model scores
# -------------------------
df = pd.read_csv(CSV_PATH)
assert {"Model", "AUROC", "AUPRC"}.issubset(df.columns), "CSV must contain Model, AUROC, and AUPRC columns."

# -------------------------
# 1. AUROC vs AUPRC scatter
# -------------------------
plt.figure(figsize=(10, 6))
plt.scatter(df["AUROC"], df["AUPRC"], color='teal', s=100)
for i, row in df.iterrows():
    plt.text(row["AUROC"] + 0.0005, row["AUPRC"], row["Model"], fontsize=9)
plt.xlabel("AUROC")
plt.ylabel("AUPRC")
plt.title("Model Predictive Performance")
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_DIR / "model_performance_scatter.png")
plt.close()

# -------------------------
# 2. AUPRC bar chart
# -------------------------
plt.figure(figsize=(10, 6))
plt.barh(df["Model"], df["AUPRC"], color='orange', edgecolor='black')
plt.xlabel("AUPRC")
plt.title("Model Evaluation – AUPRC")
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(SAVE_DIR / "model_auprc_barchart.png")
plt.close()

# -------------------------
# 3. Highlight best AUPRC
# -------------------------
best_model = df.loc[df["AUPRC"].idxmax()]
colors = ['green' if model == best_model["Model"] else 'gray' for model in df["Model"]]

plt.figure(figsize=(10, 6))
plt.barh(df["Model"], df["AUPRC"], color=colors, edgecolor='black')
plt.xlabel("AUPRC")
plt.title("Optimal Model Selection (Best AUPRC Highlighted)")
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(SAVE_DIR / "model_best_auprc_highlighted.png")
plt.close()

print(f"✅ Plots saved to {SAVE_DIR.resolve()}")
