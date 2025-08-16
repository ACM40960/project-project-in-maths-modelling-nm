import lightgbm as lgb
import pandas as pd
from pathlib import Path

# -----------------------
# Config
# -----------------------
MODEL_PATH = Path("models/booster_0.txt")   # trained LightGBM model
OUTPUT_CSV = Path("models/feature_importance_top10.csv")

# -----------------------
# Load the model
# -----------------------
booster = lgb.Booster(model_file=str(MODEL_PATH))

# -----------------------
# Extract feature importance (Gain)
# -----------------------
feature_names = booster.feature_name()
gain_importance = booster.feature_importance(importance_type="gain")

df_importance = pd.DataFrame({
    "Feature": feature_names,
    "GainImportance": gain_importance
}).sort_values(by="GainImportance", ascending=False).reset_index(drop=True)

# -----------------------
# Keep top 10, group others
# -----------------------
if len(df_importance) > 10:
    top10 = df_importance.iloc[:10].copy()
    others_sum = df_importance.iloc[10:]["GainImportance"].sum()
    top10.loc[len(top10)] = ["Others", others_sum]
else:
    top10 = df_importance.copy()

# Save
top10.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Top 10 + Others feature importance saved to: {OUTPUT_CSV.resolve()}")
print(top10)
