import pandas as pd
import os
import argparse
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

def preprocess_seed(path):
    df = pd.read_csv(path)
    # Convert to categorical where appropriate
    for col in ['customer', 'merchant', 'gender', 'category']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print(f"[INFO] Seed data shape: {df.shape}")
    return df

def create_metadata(df, table_name='transactions'):
    metadata = Metadata.detect_from_dataframe(data=df, table_name=table_name)
    return metadata

def train_synthesizer(df, metadata):
    synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
    print("[INFO] Fitting synthesizer...")
    synthesizer.fit(df)
    return synthesizer

def generate_data(synthesizer, num_rows):
    print(f"[INFO] Sampling {num_rows} rows of synthetic data...")
    return synthesizer.sample(num_rows)

def save_df(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Synthetic CSV saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to seed CSV")
    parser.add_argument("--output", default="data/synthetic_txns.csv", help="Output CSV path")
    parser.add_argument("--rows", type=int, default=10000, help="Number of synthetic rows to generate")
    args = parser.parse_args()

    df = preprocess_seed(args.input)
    metadata = create_metadata(df)
    synthesizer = train_synthesizer(df, metadata)
    synthetic_df = generate_data(synthesizer, args.rows)
    save_df(synthetic_df, args.output)
