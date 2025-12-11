import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="NFL Betting Model",
    layout="wide"
)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("nfl_2025_edges_with_model.csv")
    except Exception as e:
        st.error(f"Error loading nfl_2025_edges_with_model.csv: {e}")
        return None

    # Basic required columns check
    required_cols = ["home_team", "away_team"]
    for c in required_cols:
        if c not in df.columns:
            st.error(f"Missing required column in CSV: {c}")
            return None

    # Ensure week column exists
    if "week" not in df.columns:
        df["week"] = 0

    # Compute implied_home_prob if missing
    if "implied_home_prob" not in df.columns and "home_market_ml" in df.columns:
        def ml_to_prob(ml):
            try:
                ml = float(ml)
            except (TypeError, ValueError):
                return np.nan
            if ml > 0:
                return 100 / (ml + 100)
            else:
                return -ml / (-ml + 100)
        df["implied_home_prob"] = df["home_market_ml"].apply(ml_to_prob)

    # Compute edge if missing and we have model_home_prob
    if "edge" not in df.columns and "model_home_prob" in df.columns and "implied_home_prob" in df.columns:
        df["edge"] = df["model_home_prob"] - df["implied_home_prob"]

    # If bet_units missing, set to 0
    if "bet_units" not in df.columns:
        df["bet_units"] = 0.0

    # Default recommend_bet if missing
    if "recommend_bet" not in df.columns and "edge" in df.columns:
        df["recommend_bet"] = (df["edge"] > 0.04) & (df["bet_units"] >= 0.25)

    return df


df = load_data()
if df is None:
    st.stop()

st.title("🏈 NFL Betting Model Dashboard")

# ---------- SIDEBAR FILTERS ----------
st.sidebar.header("Filters")

weeks = sorted(df["week"].dropna().unique().tolist())
if len(weeks) > 0:
    selected_week = st.sidebar.selectbox("Week", options=["All"] + weeks, index=0)
else:
    selected_week = "All"

teams = sorted(set(df["home_team"]).union(set(df["away_team"])))
selected_teams = st.sidebar.multiselect("Filter by team (home or away)", options=teams, default=[])

df_filtered = df.copy()

if selected_week != "All":
    df_filtered = df_filtered[df_filtered["week"] == selected_week]

if selected_teams:
    df_filtered = df_filtered[
        df_filtered["home_team"].isin(selected_teams)
        | df_filtered["away_team"].isin(selected_teams)
    ]

# ---------- MAIN GAMES TABLE ----------
st.header("📊 All Games")

# Columns we *try* to show
main_cols = [
    "week",
    "home_team",
    "away_team",
    "home_market_ml",
    "home_market_spread",
    "model_home_prob",
    "implied_home_prob",
    "edge",
    "model_margin",
    "model_total_points",
    "bet_units",
]

main_cols = [c for c in main_cols if c in df_filtered.columns]

if df_filtered.empty:
    st.info("No games match your current filters.")
else:
    table = df_filtered[main_cols].copy()

    # Pretty formatting
    if "model_home_prob" in table.columns:
        table["model_home_prob"] = (table["model_home_prob"] * 100).round(1).astype(str) + "%"
    if "implied_home_prob" in table.columns:
        table["implied_home_prob"] = (table["implied_home_prob"] * 100).round(1).astype(str) + "%"
    if "edge" in table.columns:
        table["edge"] = (table["edge"] * 100).round(2).astype(str) + "%"
    if "bet_units" in table.columns:
        table["bet_units"] = table["bet_units"].round(2)

    st.dataframe(table, use_container_width=True)

# ---------- RECOMMENDED BETS (KELLY) ----------
st.header("📈 Recommended Bets (Kelly Sizing)")

bets = df.copy()

# If implied prob still missing in original df, compute it quickly
if "implied_home_prob" not in bets.columns and "home_market_ml" in bets.columns:
    def ml_to_prob2(ml):
        try:
            ml = float(ml)
        except (TypeError, ValueError):
            return np.nan
        if ml > 0:
            return 100 / (ml + 100)
        else:
            return -ml / (-ml + 100)
    bets["implied_home_prob"] = bets["home_market_ml"].apply(ml_to_prob2)

# UI controls
col1, col2, col3 = st.columns(3)
with col1:
    min_edge_pct = st.slider("Minimum edge (%)", 0.0, 15.0, 4.0, 0.5)
with col2:
    min_units = st.slider("Minimum bet size (units)", 0.0, 5.0, 0.25, 0.25)
with col3:
    this_week_only = st.checkbox("Show this week only", value=True)

bets_rec = bets.copy()

# Only rows that are recommended if column exists
if "recommend_bet" in bets_rec.columns:
    bets_rec = bets_rec[bets_rec["recommend_bet"]]

# Filter by edge & units
if "edge" in bets_rec.columns:
    bets_rec = bets_rec[bets_rec["edge"] * 100 >= min_edge_pct]
if "bet_units" in bets_rec.columns:
    bets_rec = bets_rec[bets_rec["bet_units"] >= min_units]

# Filter to current week if requested
if this_week_only and "week" in bets_rec.columns and len(weeks) > 0:
    current_week = max(weeks)
    bets_rec = bets_rec[bets_rec["week"] == current_week]

# Sort by biggest edge
if "edge" in bets_rec.columns:
    bets_rec = bets_rec.sort_values("edge", ascending=False)

if bets_rec.empty:
    st.info("No bets meet your current edge / unit criteria.")
else:
    rec_cols = [
        "week",
        "home_team",
        "away_team",
        "home_market_ml",
        "home_market_spread",
        "model_home_prob",
        "implied_home_prob",
        "edge",
        "bet_units",
    ]
    rec_cols = [c for c in rec_cols if c in bets_rec.columns]
    rec_table = bets_rec[rec_cols].copy()

    # Pretty formatting
    if "model_home_prob" in rec_table.columns:
        rec_table["model_home_prob"] = (rec_table["model_home_prob"] * 100).round(1).astype(str) + "%"
    if "implied_home_prob" in rec_table.columns:
        rec_table["implied_home_prob"] = (rec_table["implied_home_prob"] * 100).round(1).astype(str) + "%"
    if "edge" in rec_table.columns:
        rec_table["edge"] = (rec_table["edge"] * 100).round(2).astype(str) + "%"
    if "bet_units" in rec_table.columns:
        rec_table["bet_units"] = rec_table["bet_units"].round(2)

    st.dataframe(rec_table, use_container_width=True)

st.caption("Model outputs are for informational purposes only and do not guarantee future results.")
