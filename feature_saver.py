import json, pathlib, pandas as pd
from banksim_fraud.features import engineer, feature_list
from banksim_fraud.config   import MODEL_DIR

# --- point to your training CSV
csv_path = "data/bs140513_032310.csv"

df  = engineer(pd.read_csv(csv_path))
feat_schema = feature_list(df)

(MODEL_DIR).mkdir(exist_ok=True)
(MODEL_DIR / "features.json").write_text(json.dumps(feat_schema))

print(f"Wrote schema ({len(feat_schema)} cols) → {MODEL_DIR/'features.json'}")
