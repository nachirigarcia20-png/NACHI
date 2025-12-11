import streamlit as st
import pandas as pd
import requests
import os

# --------------------------
# STREAMLIT PAGE SETTINGS
# --------------------------
st.set_page_config(page_title="NFL Betting Model", layout="wide")

st.title("🏈 NFL AI Betting Model")
st.caption("Real 2025 season from CSV + live odds from TheOddsAPI.")


# --------------------------
# LOAD MODEL EDGES FROM CSV
# --------------------------
@st.cache_data
def load_model_edges():
    try:
        # Always look for CSV in SAME FOLDER as app.py
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "model_edges_2025.csv")

        st.write(f"DEBUG – Looking for CSV at: `{csv_path}`")

        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        st.error("❌ model_edges_2025.csv was NOT found in the same folder as app.py.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error reading model_edges_2025.csv: {e}")
        return pd.DataFrame()


edges_df = load_model_edges()

st.write("### DEBUG – Columns in your CSV")
st.write(list(edges_df.columns) if not edges_df.empty else "No data loaded")

required_cols = [
    "week",
    "home_team",
    "away_team",
    "model_home_win_prob",
    "model_away_win_prob",
    "edge_spread",
    "edge_moneyline",
]

if edges_df.empty:
    st.stop()

missing = [c for c in required_cols if c not in edges_df.columns]
if missing:
    st.error(f"❌ CSV is missing columns: {missing}")
    st.stop()

# --------------------------
# WEEK SELECTOR + MODEL TABLE
# --------------------------
weeks = sorted(edges_df["week"].astype(int).unique().tolist())
selected_week = st.sidebar.selectbox("Select NFL Week", weeks, index=0)

week_df = edges_df[edges_df["week"] == selected_week].copy()

st.subheader(f"📊 Model Edges – Week {selected_week}")
model_cols = [
    "home_team",
    "away_team",
    "model_home_win_prob",
    "model_away_win_prob",
    "edge_spread",
    "edge_moneyline",
]
st.dataframe(week_df[model_cols], use_container_width=True)


# --------------------------
# FETCH LIVE ODDS (THEODDSAPI)
# --------------------------
@st.cache_data(ttl=60 * 5)
def fetch_live_odds():
    odds_key = st.secrets["ODDS_API_KEY"]

    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {
        "apiKey": odds_key,
        "regions": "us",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for game in data:
        home = game.get("home_team")
        away = game.get("away_team")
        home_ml = away_ml = None
        home_spread = away_spread = None

        bookmakers = game.get("bookmakers") or []
        if not bookmakers:
            continue

        bm = bookmakers[0]
        for market in bm.get("markets", []):
            if market.get("key") == "h2h":
                for o in market.get("outcomes", []):
                    if o.get("name") == home:
                        home_ml = o.get("price")
                    elif o.get("name") == away:
                        away_ml = o.get("price")
            if market.get("key") == "spreads":
                for o in market.get("outcomes", []):
                    if o.get("name") == home:
                        home_spread = o.get("point")
                    elif o.get("name") == away:
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

st.markdown("---")
st.subheader("💰 Live Moneyline & Spread – Edges For This Week")

try:
    odds_df = fetch_live_odds()
    if odds_df.empty:
        st.warning("No odds data returned from TheOddsAPI.")
    else:
        merged = week_df.merge(
            odds_df,
            on=["home_team", "away_team"],
            how="left",
        )
        if "edge_spread" in merged.columns:
            merged = merged.sort_values("edge_spread", ascending=False)
        st.dataframe(merged, use_container_width=True)
except Exception as e:
    st.error(f"❌ Could not fetch or merge live odds: {e}")


# --------------------------
# HEAD-TO-HEAD MATCHUP (MODEL ONLY)
# --------------------------
st.markdown("---")
st.subheader("🤝 Head-to-Head Matchup (Model Only)")

teams = sorted(pd.unique(edges_df[["home_team", "away_team"]].values.ravel()))
col1, col2 = st.columns(2)
with col1:
    home_sel = st.selectbox("Home team", teams, index=0)
with col2:
    away_sel = st.selectbox("Away team", teams, index=1 if len(teams) > 1 else 0)

if st.button("Predict this matchup"):
    if home_sel == away_sel:
        st.warning("Home and away teams must be different.")
    else:
        mask1 = (edges_df["home_team"] == home_sel) & (edges_df["away_team"] == away_sel)
        mask2 = (edges_df["home_team"] == away_sel) & (edges_df["away_team"] == home_sel)

        if edges_df[mask1].shape[0] > 0:
            row = edges_df[mask1].iloc[0]
            home_prob = row["model_home_win_prob"]
            away_prob = row["model_away_win_prob"]
        elif edges_df[mask2].shape[0] > 0:
            row = edges_df[mask2].iloc[0]
            home_prob = row["model_away_win_prob"]
            away_prob = row["model_home_win_prob"]
        else:
            st.error("This matchup is not in your data yet.")
            home_prob = away_prob = None

        if home_prob is not None:
            st.write(f"**Model win probability {home_sel} (home): {home_prob:.1%}**")
            st.write(f"**Model win probability {away_sel} (away): {away_prob:.1%}**")
