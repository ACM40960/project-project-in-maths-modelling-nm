import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

DATA_PATH = "data/live_scored_txns.csv"

st.set_page_config(layout="wide")
st.title("BankSim Live Fraud Detection Dashboard")

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
            st.subheader("Summary Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", f"{len(df):,}")
            col2.metric("Fraudulent Transactions", int(df["fraud"].sum()))
            col3.metric("Fraud Rate", f"{df['fraud'].mean():.2%}")

            st.markdown("---")

            recent_df = df.tail(300)
            col4, col5 = st.columns([2, 3])

            # Fraud by category (Coral Green)
            with col4:
                st.caption("Fraud Count by Category (Last 300 Txns)")
                cat_fraud = recent_df[recent_df["fraud"] == 1]["category"].value_counts().sort_values()
                fig1, ax1 = plt.subplots(figsize=(5, 3))
                ax1.barh(cat_fraud.index.astype(str), cat_fraud.values, color="#3CB371")  # Coral green
                ax1.set_xlabel("Count")
                ax1.set_ylabel("Category")
                ax1.tick_params(labelsize=8)
                fig1.tight_layout()
                st.pyplot(fig1)

            # Rolling fraud rate (Royal Blue)
            with col5:
                st.caption("Rolling Fraud Rate (Window=50)")
                df_time = recent_df[["step", "fraud"]].copy().sort_values("step")
                df_time["rolling_rate"] = df_time["fraud"].rolling(50, min_periods=1).mean()
                fig3, ax3 = plt.subplots(figsize=(7, 3))
                ax3.plot(df_time["step"], df_time["rolling_rate"], color="#4169E1", linewidth=2)
                ax3.set_xlabel("Step")
                ax3.set_ylabel("Rolling Fraud Rate")
                ax3.grid(True, linestyle="--", alpha=0.5)
                fig3.tight_layout()
                st.pyplot(fig3)

            st.markdown("---")
            st.subheader("Model Confidence (Fraud Probability)")
            fig2, ax2 = plt.subplots(figsize=(7, 3))
            ax2.hist(df["probability"], bins=30, color="#4B0082", edgecolor="black")
            ax2.set_xlabel("Predicted Fraud Probability")
            ax2.set_ylabel("Frequency")
            fig2.tight_layout()
            st.pyplot(fig2)

            st.markdown("---")
            st.subheader("Recent Transactions")
            st.dataframe(df.head(25), use_container_width=True)

    else:
        st.warning("Waiting for data...")

    time.sleep(3)
