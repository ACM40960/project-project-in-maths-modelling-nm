&#x20;     &#x20;

## Table of Contents

1. [Abstract](#abstract)
2. [Project Description](#project-description)
   - [Key Components](#key-components)
   - [Imbalance Strategies](#imbalance-strategies)
   - [Project Goals](#project-goals)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Steps](#steps)
   - [Notes](#installation-notes)
5. [Data Source](#data-source)
6. [Feature Engineering](#feature-engineering)
7. [Model Training](#model-training)
   - [Quick Start](#quick-start)
   - [Custom Training](#custom-training)
8. [Evaluation](#evaluation)
9. [Deployment](#deployment)
10. [Results](#results)
11. [Contributing](#contributing)
12. [License](#license)
13. [Contact](#contact)
14. [Credits](#credits)

## Abstract

Credit‑card fraud is rare—less than 2 % of transactions—but costly.\
This repository demonstrates a production‑minded pipeline that ingests the open‑source **BankSim** dataset, engineers behaviour‑driven features, and trains imbalance‑aware tree ensembles (LightGBM & XGBoost) to flag suspicious swipes in near‑real time.\
The ensemble attains an **AUROC of 0.985** and retains sub‑10 ms scoring latency, making it suitable for real‑time decisioning.

## Project Description

The notebook and scripts walk through the entire ML life‑cycle—from raw CSV to a deployable micro‑service.

### Key Components

- **Data ingestion & cleansing** – robust loading, duplicate and null pruning.
- **Temporal reconstruction** – converts BankSim’s `step` counter into calendar timestamps, enabling rolling features.
- **Feature engineering** – rolling z‑scores, night‑time flags, customer‑merchant interaction counts, one‑hot categories.
- **Model zoo** – four imbalance‑aware variants:
  1. Weighted LightGBM
  2. Easy undersample (5× LGBM ensemble)
  3. SMOTE + LightGBM
  4. Weighted XGBoost
- **Hyper‑parameter optimisation** – Optuna sweeps \~50 trials for the champion model.
- **Model persistence** – boosters saved as binary `.txt` artifacts consumable by C++/Python runtimes.
- **Feature schema export** – `feature_saver.py` writes a JSON feature schema (`models/features.json`) that the inference API uses to validate incoming requests.

### Imbalance Strategies

Fraud makes up \~1.2 % of records. The repo compares cost‑sensitive weighting, undersampling, and synthetic minority oversampling (SMOTE) to mitigate bias.

### Project Goals

- Ship a **re‑usable template** for fraud modelling on tabular data.
- Maintain **explainability** via SHAP values to satisfy PSD2 & card‑scheme audit trails.
- Provide a **deployment path** (FastAPI micro‑service) with <10 ms prediction latency.

## Project Structure

```text
fraud_detection_pipeline/
├── data/                       # Raw & interim datasets
├── notebook/                   # Exploratory analysis & feature crafting
│   └── fraud_pipeline.ipynb
├── src/
│   ├── config.py               # Global paths & constants
│   ├── prepare_data.py         # Cleansing & feature engineering
│   ├── train.py                # Model training CLI
│   ├── evaluate.py             # Metric & plot generation
│   ├── serve.py                # FastAPI inference server
│   └── utils.py
├── models/                     # Saved boosters & encoders
├── requirements.txt
├── Dockerfile                  # Containerised inference image
├── .gitignore
└── README.md
```

## Installation

### Prerequisites

- Python **3.10+**
- `git`, `make`
- Optional: Docker (for containerised serving)

### Steps

```bash
# 1. Clone
git clone https://github.com/<your‑handle>/fraud_detection_pipeline.git
cd fraud_detection_pipeline

# 2. Create & activate virtual env
python -m venv venv
source venv/bin/activate          # Windows: venv\\Scripts\\activate

# 3. Install deps
pip install -r requirements.txt

# 4. Run quick demo
make demo                          # trains a weighted LGBM & prints metrics
```

### Installation Notes

- On Debian‑based Linux you may need: `sudo apt-get install build-essential libssl-dev`.
- Apple Silicon users: set `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` when using XGBoost.

#### Using Poetry (alternative)

If you prefer **Poetry** over plain `pip`, do:

```bash
pip install poetry
poetry install            # installs according to pyproject.toml
poetry run make demo      # same quick demo inside the venv
```

## Data Source

The project uses the **BankSim synthetic financial dataset** (⭳ `bs140513_032310.csv`, 600 K transactions).\
If the file is not present under `data/raw/`, run:

```bash
make fetch-data
```

to download it from the original Zenodo mirror.

## Feature Engineering

| Feature               | Type  | Description                                           |
| --------------------- | ----- | ----------------------------------------------------- |
| `amount_zscore_24h`   | float | Z‑score of transaction amount vs customer’s last 24 h |
| `txn_count_24h`       | int   | Number of customer transactions in last 24 h          |
| `merchant_occurences` | int   | Cumulative customer‑merchant pair count               |
| `is_night`            | bool  | 1 if 00:00–06:00                                      |
| `cat_(*)`             | dummy | One‑hot encoded purchase category                     |

All features are generated via `prepare_data.py` and cached under `data/processed/`.

## Model Training

### Quick Start

```bash
python src/train.py \
  --model weighted_lgbm \
  --output models/lgb_weighted.txt
```

### Custom Training

See `python src/train.py --help` for the full CLI—toggle model type, Optuna study size, early stopping, and device settings (CPU/GPU).

## Evaluation

Run

```bash
python src/evaluate.py --model models/lgb_weighted.txt
```

to generate:

- Confusion matrix (`images/confusion_matrix.png`)
- ROC & PR curves (`images/roc_curve.png`, `images/pr_curve.png`)
- Calibration plot (`images/calibration_curve.png`)







## Deployment

The repository ships a lightweight FastAPI service:

```bash
uvicorn src.serve:app --port 8000
```

`POST /predict` expects a JSON feature vector and returns fraud probability. See `serve.py` for details.\
A Dockerfile is provided for containerized deployment:

```bash
docker build -t fraud-scorer .
docker run -p 8000:8000 fraud-scorer
```

## Results

| Metric                 | Score  |
| ---------------------- | ------ |
| **AUROC**              | 0.985  |
| **AUPRC**              | 0.612  |
| **Recall @ 0.1 % FPR** | 0.74   |
| **Latency (p95)**      | 8.1 ms |



> *Numbers above are averaged over a 5‑fold temporal split.*

## Contributing

Pull requests are welcome! Open an issue first to discuss major changes.\
Please follow the conventional commits spec and run `make lint` before pushing.

## License

Distributed under the MIT License. See `LICENSE` for details.

## Contact

Maintainer – [Your Name](mailto\:your.email@example.com)

## Credits

- Original **BankSim** dataset by López‑Rojas & Axelsson (2014).
- LightGBM, XGBoost, Optuna, Imbalanced‑Learn open‑source communities.

