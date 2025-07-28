# Fraud Detection in Financial Transactions

This project builds, tunes, and evaluates multiple machine learning models for credit card fraud detection using the open-source **BankSim** dataset.  
It includes synthetic data generation, feature engineering, class imbalance strategies, and model deployment components.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Synthetic Data](#synthetic-data)
5. [Feature Engineering](#feature-engineering)
6. [Modeling Approaches](#modeling-approaches)
7. [Evaluation](#evaluation)
8. [Scoring API](#scoring-api)
9. [Live Dashboard](#live-dashboard)
10. [Results](#results)
11. [Credits](#credits)

---

## Overview

Fraud detection is a rare-event classification problem with high costs for false negatives. We build and benchmark four modeling pipelines on the **BankSim** dataset:

This project tackles financial fraud detection as a rare-event classification task using the **BankSim** dataset.  
We systematically explored and compared four modeling approaches, all grounded in domain-informed behavioral features and tailored class imbalance strategies.

---
### Highlights:
- Evaluated multiple classifiers — including Logistic Regression, Random Forest, XGBoost, and LightGBM — before finalizing **a tuned LightGBM ensemble with a 0.02 undersampling ratio**, which achieved the best balance of precision, recall, and latency.
- Engineered 39 behavioral and temporal features such as rolling z-scores, transaction velocity, and customer–merchant familiarity using BankSim’s event log format.
- Compared **four imbalance-aware techniques**:
  1. Weighted LightGBM (Optuna-tuned)
  2. Easy-negative undersampled LightGBM ensemble (×5)
  3. SMOTE + LightGBM
  4. Weighted XGBoost (Optuna-tuned)
- All models evaluated on **out-of-time splits**, using AUPRC and AUROC as primary metrics.
- Final deployment includes a real-time **FastAPI scoring endpoint** and a **Streamlit dashboard** visualizing fraud scores as transactions stream in.

This end-to-end solution demonstrates how principled feature design and imbalance handling can yield scalable, interpretable fraud detection systems that outperform generic baselines in both accuracy and responsiveness.

---

### Project Structure
```text
banksim-fraud/
├── src/
│   └── banksim_fraud/
│       ├── api.py                 # FastAPI scoring service
│       ├── features.py            # Feature generation
│       ├── model.py               # Model I/O and prediction
│       └── config.py              # Paths, threshold, etc.
├── tools/
│   ├── dashboard.py               # Streamlit monitoring UI
│   ├── stream_and_score.py        # Synthetic stream → API
│   ├── run_demo.py                # Orchestrates all components
│   └── generate_synthetic_data.py # SDV-based data synthesis
├── models/                        # Trained models and schemas
├── data/                          # Raw, processed, and scored data
├── assets/                        # Images, plots, diagrams
├── requirements.txt
└── README.md
```

### Installation
```
git clone https://github.com/yourname/banksim-fraud
cd banksim-fraud

python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

pip install -r requirements.txt

```


### Synthetic Data Generation

```
python tools/generate_synthetic_data.py \
  --input data/bs140513_032310.csv \
  --output data/synthetic_txns.csv \
  --rows 50000

```



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

## Methodology

### 1. Dataset Preparation

We used the **BankSim synthetic financial dataset**, a benchmark dataset generated using agent-based modeling to simulate realistic banking behavior.  
It contains approximately **594,643** timestamped transactions, each described by the following raw fields:

- `step`: Time (in hours) since the beginning of the simulation (1–740)
- `customer`: Unique customer identifier
- `merchant`: Unique merchant identifier
- `age`: Customer age group
- `gender`: Customer gender
- `category`: Transaction type/category
- `amount`: Monetary value of the transaction
- `fraud`: Binary label (1 = fraud, 0 = legitimate)

A sample of the first five rows is shown below:

| step | customer      | age | gender | zipcodeOri | merchant     | zipMerchant | category          | amount | fraud |
|------|---------------|-----|--------|-------------|--------------|-------------|-------------------|--------|--------|
| 0    | C1093826151   | 4   | M      | 28007       | M348934600   | 28007       | es_transportation | 4.55   | 0      |
| 0    | C352968107    | 2   | M      | 28007       | M348934600   | 28007       | es_transportation | 39.68  | 0      |
| 0    | C2054744914   | 4   | F      | 28007       | M1823072687  | 28007       | es_transportation | 26.89  | 0      |
| 0    | C1760612790   | 3   | M      | 28007       | M348934600   | 28007       | es_transportation | 17.25  | 0      |
| 0    | C757503768    | 5   | M      | 28007       | M348934600   | 28007       | es_transportation | 35.72  | 0      |

---

#### Preprocessing Steps

We performed the following steps to prepare the data for modeling:

- **Removed nulls or duplicates** (though BankSim is clean by design, this was verified)
- **Categorical encoding**:
  - Converted `customer`, `merchant`, `gender`, `category`, and `age` into encoded features
  - One-hot encoding was applied selectively (e.g., `category`, `gender`)
-  **Temporal alignment**:
  - The `step` column (1 unit = 1 hour) was retained and used to reconstruct transaction sequences
  - Used to define time-aware features like 24-hour rolling windows
-  **Train/Test split**:
  - To simulate real-time deployment, we created an **out-of-time split**:
    - Training set: transactions where `step ≤ 600` (~80%)
    - Testing set: transactions where `step > 600` (~20%)

This procedure produced a clean, fully numeric feature matrix with consistent row-level semantics: each row represents a single transaction, described by both raw and engineered features, and labeled as fraudulent or not.

---

### 2. Feature Engineering

Effective fraud detection requires features that capture **temporal behavior**, **transactional patterns**, and **entity interactions**. While the raw BankSim dataset includes only basic fields like amount, customer ID, merchant ID, and category, we engineered 39 new features to enable models to learn from behavioral signals over time.

Our feature engineering process followed these principles:

- **Temporal alignment**: Features were constructed such that only information from **prior transactions** was used — no data leakage from future activity.
- **Entity-aware**: Features were computed per **customer**, **merchant**, or **customer–merchant pair** to capture personal and network-based behavior.
- **Real-time feasible**: All features can be computed in streaming mode and used for instant scoring.

We grouped the engineered features into the following categories:

---

#### Temporal Context Features

These features model user behavior **over time**, especially within rolling windows:

| Feature                | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `amount_zscore_24h`    | Z-score of current amount vs customer’s prior 24h spending                 |
| `txn_count_24h`        | Number of transactions in last 24h (by customer)                           |
| `mean_amount_24h`      | Rolling mean of customer’s transaction amounts over past 24h              |
| `step_delta`           | Time difference (in steps) from previous transaction                      |

Purpose: capture abnormal spending, bursts, or lulls.

---

#### Customer–Merchant Interaction Features

These features describe familiarity and recurrence in customer–merchant pairings:

| Feature                   | Description                                                       |
|---------------------------|-------------------------------------------------------------------|
| `merchant_occurrences`    | Total number of times a customer has transacted with this merchant |
| `is_first_time_pair`      | Binary flag if this is the first occurrence of this pair           |
| `unique_merchants_count`  | Count of distinct merchants the customer has interacted with       |

Purpose: surface rare or novel interactions, which are often more suspicious.

---

#### Time-of-Day Features

These features flag transactions that occur during unusual hours:

| Feature       | Description                                 |
|---------------|---------------------------------------------|
| `is_night`    | 1 if transaction occurs between 00:00–06:00 |

Purpose: many fraudulent activities spike during low-traffic hours.

---

#### Categorical Encodings

The raw field `category` is one of the only true categorical fields in BankSim. It was one-hot encoded into its different categories.
We also one-hot encoded `gender` and any other discrete fields where applicable.

#### Implementation Notes

- Engineered features were added in `features.py`, using rolling `groupby().apply()` for customer-based windows.
- Caching and memoization were used to speed up feature computation across large datasets.
- Missing values were handled by filling with 0 (for counts) or rolling medians (for stats).
- All features were standardized where appropriate to stabilize learning.



### 3. Modeling & Imbalance Handling Strategies

Fraud detection in BankSim presents a significant **class imbalance** challenge — with frauds representing only **~1.2%** of all transactions.  
To address this, we implemented four modeling pipelines, each paired with a different **imbalance mitigation strategy**, and evaluated them under consistent conditions.

All models were trained on the same feature set and validated using a strict **out-of-time split** (step > 600).  
Metrics focused on **AUPRC** (primary), **AUROC**, **Recall@FPR**, and **Precision@k**.

---

#### Strategy 1: Weighted LightGBM (Baseline)

- **Approach**: Applied class weighting using `scale_pos_weight = non-fraud / fraud`.
- **Tuning**: Hyperparameters optimized via **Optuna** (20 trials).
- **Outcome**: Delivered a strong baseline with minimal tuning effort.
- **Advantages**:
  - Preserves full data
  - Quick to train
  - Low risk of overfitting

---

#### Strategy 2: Easy-Negative Undersample Ensemble (5× LightGBM)

- **Approach**: Used `RandomUnderSampler` to downsample clear non-fraud cases to a 0.02 ratio.
- Trained **five LightGBM models**, each on a differently bootstrapped undersampled dataset.
- Predictions were **averaged** to improve stability and generalization.

- **Advantages**:
  - Boosted fraud recall and AUPRC
  - Reduces majority-class dominance
- **Tradeoff**:
  - Higher training time (5× models)
  - Some risk of discarding informative negatives

---

#### Strategy 3: SMOTE + LightGBM

- **Approach**: Used SMOTE to synthetically oversample the fraud class to 10%.
- Trained a single LightGBM on this augmented dataset.

- **Advantages**:
  - Balances the class ratio without dropping real data
- **Tradeoff**:
  - Synthetic samples can distort decision boundaries
  - Slightly lower performance than undersampled ensemble in this context

---

#### Strategy 4: Weighted XGBoost

- **Approach**: Applied `scale_pos_weight` and tuned using **Optuna** (20 trials) with `aucpr` as the objective.
- Trained a single robust XGBoost model with regularized depth.

- **Advantages**:
  - Strong generalization
  - Native support for class imbalance and boosting regularization
- **Tradeoff**:
  - Slower to train than LightGBM
  - Requires deeper parameter tuning

---

### Summary of Results

All models were benchmarked on the same engineered features and holdout set.  
The **best performing model** was the **LightGBM ensemble with a 0.02 undersampling ratio**, which achieved the highest AUPRC and stable recall across thresholds.

This strategy was ultimately chosen for deployment in the real-time scoring pipeline.


### 4. Evaluation Protocol

All models were evaluated on a **strict out-of-time split** (step > 600), simulating real-world streaming performance.

We reported the following metrics:

- **AUPRC** (primary metric due to extreme imbalance)
- **AUROC**
- **Precision@k** (top-k flagged transactions)
- **Recall @ 0.1% FPR** (fraud recall when very few false alarms allowed)

The best-performing model — **LightGBM with undersampling ratio 0.02** — was chosen for deployment.







## Contributing

Pull requests are welcome! Open an issue first to discuss major changes.\
Please follow the conventional commits spec and run `make lint` before pushing.

## License

Distributed under the MIT License. See `LICENSE` for details.

## Contact

Maintainer – Mani Jose(mailto\:manijosevg@gmail.com), Nakul Krishna(mailto\:nakulkrishna96@gmail.com)

## Credits

- Original **BankSim** dataset by López‑Rojas & Axelsson (2014).
- LightGBM, XGBoost, Optuna, Imbalanced‑Learn open‑source communities.

