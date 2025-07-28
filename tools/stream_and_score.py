import pandas as pd
import requests
import time
import os
import json

INPUT_PATH = "data/synthetic_txns.csv"
OUTPUT_PATH = "data/live_scored_txns.csv"
API_URL = "http://localhost:8000/score"  # make sure FastAPI is running

def stream_and_score():
    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Input file not found: {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)

    # Ensure required fields exist
    required_fields = [
        "step", "customer", "merchant", "age", "gender",
        "category", "amount"
    ]

    if not all(field in df.columns for field in required_fields):
        print(f"[ERROR] Dataset is missing required fields.")
        return

    results = []

    for i, row in df.iterrows():
        try:
            # Prepare input payload
            record = {key: row[key] for key in required_fields}
            record["zipcodeOri"] = 0
            record["zipMerchant"] = 0

            # Optional: include fraud if present
            if "fraud" in row:
                record["fraud"] = int(row["fraud"])

            # Send to API
            response = requests.post(API_URL, json=record)
            response.raise_for_status()
            prediction = response.json()

            # Add prediction result to row
            record.update(prediction)
            results.append(record)

            # Save results to live-scoring output file
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

            print(f"[✓] Txn {i+1}/{len(df)} | Fraud: {prediction['fraud_flag']} | Prob: {prediction['probability']:.4f}")
            time.sleep(0.5)

        except Exception as e:
            print(f"[ERROR] Row {i+1}: {e}")
            continue

if __name__ == "__main__":
    stream_and_score()
