import streamlit as st
import pandas as pd
import numpy as np

# --------------------------
# Load data
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("nfl_2025_edges_with_model.csv")
    # Safety: ensure expected columns exist
    expected_cols = [
        "week", "home_team", "away_team",
        "home_score", "away_score",
        "model_home_prob", "implied_home_prob",
        "edge_home", "home_market_ml", "away_market_ml"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df

df = load_data()

st.title("🏈 NFL 2025 – Sportsbook Edge Model")
st.write("Model vs Sportsbook: shows where **your model** disagrees with the **moneyline**.")

# --------------------------
# Controls
# --------------------------
valid_weeks = sorted(df["week"].dropna().unique().tolist())
default_week = int(max(valid_weeks)) if valid_weeks else 1

selected_week = st.selectbox(
    "Select week",
    options=valid_weeks,
    index=valid_weeks.index(default_week) if valid_weeks else 0,
)

edge_threshold = st.slider(
    "Minimum edge to show (in %)",
    min_value=1,
    max_value=15,
    value=5,
    step=1,
)

edge_thr_decimal = edge_threshold / 100.0

# --------------------------
# Helper to label bets
# --------------------------
def decide_side(row, thr):
    edge = row["edge_home"]
    if pd.isna(edge):
        return "No odds"
    if edge >= thr:
        return f"{row['home_team']} ML"
    elif edge <= -thr:
        return f"{row['away_team']} ML"
    else:
        return "No Bet"

# --------------------------
# Filter for selected week
# --------------------------
week_df = df[df["week"] == selected_week].copy()

# Only games where we have model + implied probs
week_df = week_df[week_df["model_home_prob"].notna()]

week_df["bet_recommendation"] = week_df.apply(
    decide_side,
    axis=1,
    thr=edge_thr_decimal,
)

week_df["model_home_prob_%"] = (week_df["model_home_prob"] * 100).round(1)
week_df["implied_home_prob_%"] = (week_df["implied_home_prob"] * 100).round(1)
week_df["edge_%"] = (week_df["edge_home"] * 100).round(1)

# Separate upcoming vs past (based on score existing or not)
past_games = week_df[week_df["home_score"].notna()].copy()
upcoming_games = week_df[week_df["home_score"].isna()].copy()

# --------------------------
# Upcoming games – what to bet this week
# --------------------------
st.header(f"📅 Week {selected_week} – Upcoming games & edges")

if upcoming_games.empty:
    st.info("No upcoming games with odds and model edges for this week yet.")
else:
    show_cols = [
        "away_team", "home_team",
        "home_market_ml", "away_market_ml",
        "model_home_prob_%", "implied_home_prob_%",
        "edge_%", "bet_recommendation"
    ]
    st.subheader("Suggested bets (based on edge vs sportsbook)")
    st.dataframe(upcoming_games[show_cols].sort_values("edge_%", ascending=False))

# --------------------------
# Past games – how the model would have done
# --------------------------
st.header(f"📊 Week {selected_week} – Past games (for backtesting)")

if past_games.empty:
    st.info("No completed games with model edges for this week yet.")
else:
    past_cols = [
        "away_team", "home_team",
        "away_score", "home_score",
        "home_market_ml",
        "model_home_prob_%", "implied_home_prob_%",
        "edge_%", "bet_recommendation"
    ]
    st.dataframe(past_games[past_cols].sort_values("edge_%", ascending=False))

st.caption("Edge = model_home_prob − implied_home_prob (from sportsbook moneyline).")


