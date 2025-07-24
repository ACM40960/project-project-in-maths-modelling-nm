import click, pandas as pd, json, numpy as np
from .features import engineer, feature_list
from .model import train_ensemble, choose_threshold, save, load, classify
from .config import DEFAULT_THRESHOLD

@click.group()
def cli(): ...

# ----------------------------------------------------------------
@cli.command(help="Train ensemble and save to models/")
@click.argument("csv", type=click.Path(exists=True))
def train(csv):
    df = engineer(pd.read_csv(csv))
    X, y = df[feature_list(df)], df["fraud"].astype(int)

    # simple chronological split: last 20 % rows for validation
    split = int(len(df) * 0.8)
    boosters = train_ensemble(X.iloc[:split], y.iloc[:split])
    thr = choose_threshold(boosters, X.iloc[split:], y.iloc[split:])
    save(boosters, thr)
    click.echo(f"✅ ensemble saved. threshold = {thr:.3f}")

# ----------------------------------------------------------------
@cli.command(help="Score a single JSON row on stdin or --file")
@click.option("--file", "json_file", type=click.Path(), help="JSON lines file")
def score(json_file):
    boosters, thr = load()
    def handle(d):
        row = engineer(pd.DataFrame([d])).iloc[0]
        vec = row[feature_list(row.to_frame().T)].to_numpy(dtype=float)
        lbl, p = classify(boosters, thr, vec)
        out = {"prob": p, "fraud_flag": int(lbl)}
        click.echo(json.dumps(out))
    if json_file:
        with open(json_file) as f:
            for line in f:
                handle(json.loads(line))
    else:
        handle(json.loads(click.get_text_stream("stdin").read().strip()))
