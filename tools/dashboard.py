import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

DATA_PATH = "data/live_scored_txns.csv"

st.set_page_config(layout="wide")
st.title("Live BankSim Simulation Fraud Dashboard")

placeholder = st.empty()

while True:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)

        if 'fraud' not in df.columns or 'probability' not in df.columns:
            st.warning("Scoring data missing 'fraud' or 'probability' columns.")
            time.sleep(2)
            continue

        df = df.sort_values("step", ascending=False)
        df["fraud"] = df["fraud"].astype(int)

        with placeholder.container():
            st.markdown("### 🔍 Summary Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", len(df))
            col2.metric("Fraudulent Transactions", df["fraud"].sum())
            col3.metric("Fraud Rate", f"{df['fraud'].mean():.2%}")

            st.divider()

            # 📊 Animated-like fraud bar chart
            st.markdown("### 📊 Fraud by Category (Rolling Window)")
            rolling_df = df.tail(300)
            cat_fraud = rolling_df[rolling_df["fraud"] == 1]["category"].value_counts().sort_values()

            fig1, ax1 = plt.subplots(figsize=(6, 3))
            bars = cat_fraud.index.astype(str)
            values = cat_fraud.values
            ax1.barh(bars, values, color="orange")
            ax1.set_title("Fraud Count by Category (Last 300 txns)", fontsize=12)
            ax1.set_xlabel("Count", fontsize=10)
            ax1.tick_params(axis='both', labelsize=9)
            fig1.tight_layout()
            st.pyplot(fig1)

            # 📈 Rolling fraud rate line chart
            st.markdown("### 📈 Rolling Fraud Rate (Last 300 Txns)")
            df_time = rolling_df[["step", "fraud"]].copy().sort_values("step")
            df_time["rolling_rate"] = df_time["fraud"].rolling(50, min_periods=1).mean()
            st.line_chart(df_time.set_index("step")["rolling_rate"])

            # Model confidence histogram
            st.markdown("### 🧠 Fraud Probability Distribution")
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.hist(df["probability"], bins=30, color="skyblue", edgecolor="black")
            ax2.set_title("Fraud Probability Histogram", fontsize=12)
            ax2.set_xlabel("Probability")
            ax2.set_ylabel("Count")
            fig2.tight_layout()
            st.pyplot(fig2)

            st.divider()
            st.markdown("### 🔎 Recent Transactions")
            st.dataframe(df.head(25), use_container_width=True)

    else:
        st.warning("Waiting for data...")
    time.sleep(3)
