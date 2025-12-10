import streamlit as st
import pandas as pd
import numpy as np
import requests

# ---------- PAGE SETTINGS ----------
st.set_page_config(
    page_title="NFL Betting Edge Model",
    page_icon="🏈",
    layout="wide",
)

st.title("🏈 NFL AI Model – Edges & Live Odds")
st.caption("Model edges using this season only, plus live odds from TheOddsAPI.")

# ---------- LOAD MODEL EDGES ----------
@st.cache_data
def load_model_edges():
    """
    Load your pre-computed model outputs for this season.
    The CSV must have columns:
    week, home_team, away_team,
    model_home_win_prob, model_away_win_prob,
    edge_spread, edge_moneyline
    """
    df = pd.read_csv("model_edges_2025.csv")
    return df


# ---------- FETCH LIVE ODDS FROM THEODDSAPI ----------
@st.cache_data(ttl=60 * 5)  # cache for 5 minutes
def fetch_live_odds():
    """
    Fetch live moneyline and spread odds from TheOddsAPI
    and return a DataFrame with one row per game:
    home_team, away_team,
    market_home_ml, market_away_ml,
    market_home_spread, market_away_spread
    """
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]  # <--- comes from Streamlit secrets

    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    rows = []

    for game in data:
        home = game.get("home_team")
        away = game.get("away_team")

        home_ml = None
        away_ml = None
        home_spread = None
        away_spread = None

        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue

        # Just use first bookmaker. You can filter for a specific one if you want.
        bm = bookmakers[0]

        for m in bm.get("markets", []):
            key = m.get("key")
            if key == "h2h":
                for o in m.get("outcomes", []):
                    name = o.get("name")
                    if name == home:
                        home_ml = o.get("price")
                    elif name == away:
                        away_ml = o.get("price")
            elif key == "spreads":
                for o in m.get("outcomes", []):
                    name = o.get("name")
                    if name == home:
                        home_spread = o.get("point")
                    elif name == away:
                        away_spread = o.get("point")

        rows.append(
            {
                "home_team": home,
                "away_team": away,
                "market_home_ml": home_ml,
                "market_away_ml": away_ml,
                "market_home_spread": home_spread,
                "market_away_spread": away_spread,
            }
        )

    return pd.DataFrame(rows)


# ---------- MAIN APP ----------
edges_df = load_model_edges()

# Sidebar – choose week
available_weeks = sorted(edges_df["week"].unique())
default_week = max(available_weeks)
selected_week = st.sidebar.selectbox("Select NFL week", available_weeks, index=available_weeks.index(default_week))

week_df = edges_df[edges_df["week"] == selected_week].copy()

st.subheader(f"Model Edges – Week {selected_week}")
st.caption("These are your model outputs only (no odds yet).")

# Show main model table
model_cols_to_show = [
    "home_team",
    "away_team",
    "model_home_win_prob",
    "model_away_win_prob",
]
if "edge_spread" in week_df.columns:
    model_cols_to_show.append("edge_spread")
if "edge_moneyline" in week_df.columns:
    model_cols_to_show.append("edge_moneyline")

st.dataframe(week_df[model_cols_to_show], use_container_width=True)


# ---------- LIVE MONEYLINE & SPREAD EDGES ----------
st.markdown("---")
st.subheader("Live Moneyline & Spread – Edges For This Week")
st.caption(f"Showing edges for week {selected_week} upcoming games.")

try:
    odds_df = fetch_live_odds()

    # Merge model edges with odds on home + away team
    merged = week_df.merge(
        odds_df,
        on=["home_team", "away_team"],
        how="left",
        validate="one_to_one",
    )

    # Sort by biggest spread edge if available
    if "edge_spread" in merged.columns:
        merged = merged.sort_values("edge_spread", ascending=False)

    st.dataframe(merged, use_container_width=True)

except Exception as e:
    st.error(f"Could not fetch or merge live odds: {e}")


# ---------- HEAD-TO-HEAD MATCHUP (MODEL ONLY) ----------
st.markdown("---")
st.subheader("Head-to-Head Matchup (Model Only)")

teams = sorted(
    pd.unique(
        edges_df[["home_team", "away_team"]].values.ravel()
    )
)

col1, col2 = st.columns(2)
with col1:
    home_sel = st.selectbox("Home team", teams, index=0)
with col2:
    away_sel = st.selectbox("Away team", teams, index=0)

if st.button("Predict this matchup"):
    if home_sel == away_sel:
        st.warning("Home and away teams must be different.")
    else:
        # Look for a row with this matchup in any week
        mask1 = (edges_df["home_team"] == home_sel) & (edges_df["away_team"] == away_sel)
        mask2 = (edges_df["home_team"] == away_sel) & (edges_df["away_team"] == home_sel)

        if edges_df[mask1].shape[0] > 0:
            row = edges_df[mask1].iloc[0]
            home_prob = row["model_home_win_prob"]
            away_prob = row["model_away_win_prob"]
            st.write(f"**Model win probability {home_sel} (home): {home_prob:.1%}**")
            st.write(f"**Model win probability {away_sel} (away): {away_prob:.1%}**")
        elif edges_df[mask2].shape[0] > 0:
            # Flip if stored reversed
            row = edges_df[mask2].iloc[0]
            home_prob = row["model_away_win_prob"]
            away_prob = row["model_home_win_prob"]
            st.write(f"**Model win probability {home_sel} (home): {home_prob:.1%}**")
            st.write(f"**Model win probability {away_sel} (away): {away_prob:.1%}**")
        else:
            st.error("This matchup is not in your data yet.")

