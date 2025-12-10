import streamlit as st
import pandas as pd
import numpy as np
import requests

# --------------------------
# STREAMLIT PAGE SETTINGS
# --------------------------
st.set_page_config(page_title="NFL Betting Model", layout="wide")

st.title("🏈 NFL AI Betting Model – 2025 Season")
st.caption("Live odds + model edge system")


# --------------------------
# LOAD MODEL CSV
# --------------------------
@st.cache_data
def load_model_edges():
    """
    Load your model CSV. If the file is missing or the columns are wrong,
    the app will not crash — it will use fallback example data.
    """
    try:
        df = pd.read_csv("model_edges_2025.csv")
        return df
    except FileNotFoundError:
        st.error("❌ The file model_edges_2025.csv was NOT found in your GitHub repo.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
        return pd.DataFrame()


edges_df = load_model_edges()

# SHOW CSV COLUMNS FOR DEBUGGING
st.write("### DEBUG – Columns found in your CSV:")
st.write(list(edges_df.columns))


# --------------------------
# REQUIRED COLUMNS
# --------------------------
required_cols = [
    "week",
    "home_team",
    "away_team",
    "model_home_win_prob",
    "model_away_win_prob",
    "edge_spread",
    "edge_moneyline",
]

missing_cols = [c for c in required_cols if c not in edges_df.columns]

# If CSV missing required columns → use fallback
if missing_cols:
    st.warning(f"⚠️ Your CSV is missing required columns: {missing_cols}")
    st.info("Using fallback example data so your app runs.")

    edges_df = pd.DataFrame([
        {
            "week": 1,
            "home_team": "KC",
            "away_team": "BUF",
            "model_home_win_prob": 0.61,
            "model_away_win_prob": 0.39,
            "edge_spread": 3.2,
            "edge_moneyline": 0.06,
        },
        {
            "week": 1,
            "home_team": "DAL",
            "away_team": "PHI",
            "model_home_win_prob": 0.55,
            "model_away_win_prob": 0.45,
            "edge_spread": 1.8,
            "edge_moneyline": 0.03,
        },
    ])


# --------------------------
# WEEK SELECTOR
# --------------------------
if "week" in edges_df.columns:
    available_weeks = sorted(edges_df["week"].astype(int).unique())
    selected_week = st.sidebar.selectbox("Select NFL Week", available_weeks, index=0)
    week_df = edges_df[edges_df["week"] == selected_week].copy()
else:
    selected_week = "ALL"
    week_df = edges_df.copy()
    st.warning("⚠️ No 'week' column found — showing all data.")


st.subheader(f"📊 Model Edges – Week {selected_week}")
st.dataframe(week_df, use_container_width=True)


# --------------------------
# FETCH LIVE ODDS
# --------------------------
@st.cache_data(ttl=300)
def fetch_live_odds():
    key = st.secrets["ODDS_API_KEY"]  # stored in Streamlit Secrets

    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {
        "apiKey": key,
        "regions": "us",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    rows = []

    for game in data:
        home = game.get("home_team")
        away = game.get("away_team")

        home_ml = away_ml = None
        home_spread = away_spread = None

        if not game.get("bookmakers"):
            continue

        bm = game["bookmakers"][0]  # use first sportsbook

        for market in bm["markets"]:
            if market["key"] == "h2h":
                for o in market["outcomes"]:
                    if o["name"] == home:
                        home_ml = o["price"]
                    elif o["name"] == away:
                        away_ml = o["price"]

            if market["key"] == "spreads":
                for o in market["outcomes"]:
                    if o["name"] == home:
                        home_spread = o["point"]
                    elif o["name"] == away:
                        away_spread = o["point"]

        rows.append({
            "home_team": home,
            "away_team": away,
            "market_home_ml": home_ml,
            "market_away_ml": away_ml,
            "market_home_spread": home_spread,
            "market_away_spread": away_spread,
        })

    return pd.DataFrame(rows)


st.markdown("---")
st.subheader("💰 Live Odds + Model Edges")

try:
    odds_df = fetch_live_odds()

    merged = week_df.merge(
        odds_df,
        on=["home_team", "away_team"],
        how="left",
    )

    st.dataframe(merged, use_container_width=True)

except Exception as e:
    st.error(f"❌ Could not fetch or merge live odds: {e}")
