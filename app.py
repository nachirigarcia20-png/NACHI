import streamlit as st
import pandas as pd
import numpy as np

# --------------------------
# Load data
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("nfl_2025_edges_with_model.csv")

    expected_cols = [
        "week", "home_team", "away_team",
        "home_score", "away_score",
        "model_home_prob", "implied_home_prob",
        "edge_home", "home_market_ml", "away_market_ml",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    # If game_date exists, parse it and create day_of_week
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df["day_of_week"] = df["game_date"].dt.day_name()
    else:
        df["day_of_week"] = np.nan

    return df

df = load_data()

st.title("🏈 NFL 2025 – Sportsbook Edge Model")
st.write("Model vs Sportsbook: find where your model disagrees with the moneyline and shows an **edge**.")

# --------------------------
# SIDEBAR CONTROLS
# --------------------------
st.sidebar.header("Filters")

valid_weeks = sorted(df["week"].dropna().unique().tolist())
default_week = int(max(valid_weeks)) if valid_weeks else 1

selected_week = st.sidebar.selectbox(
    "Select week",
    options=valid_weeks,
    index=valid_weeks.index(default_week) if valid_weeks else 0,
)

edge_threshold = st.sidebar.slider(
    "Minimum edge to show (in %)",
    min_value=1,
    max_value=15,
    value=5,
    step=1,
)

edge_thr_decimal = edge_threshold / 100.0

# Thursday filter (if we have day_of_week info)
day_filter = None
if df["day_of_week"].notna().any():
    thursday_only = st.sidebar.checkbox("Show only Thursday games", value=False)
    if thursday_only:
        day_filter = ["Thursday"]

# Hide games with no bet?
only_bets = st.sidebar.checkbox("Show only recommended bets", value=True)

# --------------------------
# Helper to label bets
# --------------------------
def decide_side(row, thr):
    edge = row["edge_home"]
    if pd.isna(edge) or pd.isna(row["implied_home_prob"]):
        return "No odds"
    if edge >= thr:
        return f"{row['home_team']} ML"
    elif edge <= -thr:
        return f"{row['away_team']} ML"
    else:
        return "No Bet"

# Row styling
def style_bet_rows(row):
    rec = row["bet_recommendation"]
    if "ML" in str(rec):
        # Strong bet recommendation → light green background
        return ["background-color: rgba(0, 200, 0, 0.15)"] * len(row)
    elif rec == "No Bet":
        # No Bet → light gray background
        return ["background-color: rgba(150, 150, 150, 0.10)"] * len(row)
    else:
        return [""] * len(row)

# --------------------------
# Filter for selected week
# --------------------------
week_df = df[df["week"] == selected_week].copy()

# If Thursday-only is on and we have day info
if day_filter is not None:
    week_df = week_df[week_df["day_of_week"].isin(day_filter)]

# We can only use rows with model prob
week_df = week_df[week_df["model_home_prob"].notna()].copy()

# Calculate bet recommendation
week_df["bet_recommendation"] = week_df.apply(
    decide_side,
    axis=1,
    thr=edge_thr_decimal,
)

week_df["model_home_prob_%"] = (week_df["model_home_prob"] * 100).round(1)
week_df["implied_home_prob_%"] = (week_df["implied_home_prob"] * 100).round(1)
week_df["edge_%"] = (week_df["edge_home"] * 100).round(1)

# Separate upcoming vs past based on scores
upcoming_games = week_df[week_df["home_score"].isna()].copy()
past_games = week_df[week_df["home_score"].notna()].copy()

if only_bets:
    upcoming_games = upcoming_games[upcoming_games["bet_recommendation"].str.contains("ML", na=False)]
    past_games = past_games[past_games["bet_recommendation"].str.contains("ML", na=False)]

# --------------------------
# Upcoming games – what to bet
# --------------------------
st.header(f"📅 Week {selected_week} – Upcoming games & edges")

if upcoming_games.empty:
    st.info("No upcoming games with odds and edges for this selection.")
else:
    show_cols_upcoming = [
        "away_team", "home_team",
        "home_market_ml", "away_market_ml",
        "model_home_prob_%", "implied_home_prob_%",
        "edge_%", "bet_recommendation",
    ]

    st.subheader("Suggested bets (based on edge vs sportsbook)")
    styled_upcoming = (
        upcoming_games[show_cols_upcoming]
        .sort_values("edge_%", ascending=False)
        .style.apply(style_bet_rows, axis=1)
    )
    st.dataframe(styled_upcoming, use_container_width=True)

# --------------------------
# Past games – backtest
# --------------------------
st.header(f"📊 Week {selected_week} – Completed games (backtest)")

if past_games.empty:
    st.info("No completed games with model edges for this selection.")
else:
    show_cols_past = [
        "away_team", "home_team",
        "away_score", "home_score",
        "home_market_ml",
        "model_home_prob_%", "implied_home_prob_%",
        "edge_%", "bet_recommendation",
    ]

    styled_past = (
        past_games[show_cols_past]
        .sort_values("edge_%", ascending=False)
        .style.apply(style_bet_rows, axis=1)
    )
    st.dataframe(styled_past, use_container_width=True)

st.caption("Edge = model_home_prob − implied_home_prob (from sportsbook moneyline). Green rows = bet; gray rows = pass.")
