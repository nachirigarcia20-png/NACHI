import streamlit as st
import pandas as pd
import numpy as np
import requests
import re

# ------------------------------------
# STREAMLIT PAGE CONFIG
# ------------------------------------
st.set_page_config(page_title="NFL AI Betting Model", layout="wide")
st.title("🏈 NFL AI Betting Model – Schedule + Live Odds")
st.caption("Schedule from RapidAPI + live odds from TheOddsAPI + your model edges.")


# ------------------------------------
# CONFIG
# ------------------------------------
SEASON = 2025            # change if you want another season
NFL_LEAGUE_ID = "1"      # NFL league id for the american-football API (check docs if needed)

RAPIDAPI_HOST_DEFAULT = "api-american-football.p.rapidapi.com"


# ------------------------------------
# HELPERS
# ------------------------------------
def get_secret(name: str, default: str | None = None) -> str | None:
    """Safe access to Streamlit secrets."""
    try:
        return st.secrets[name]
    except Exception:
        return default


# ------------------------------------
# FETCH SCHEDULE FROM RAPIDAPI
# ------------------------------------
@st.cache_data(ttl=60 * 60)
def fetch_schedule_from_rapidapi(season: int = SEASON) -> pd.DataFrame:
    """
    Fetch NFL games from RapidAPI (american-football API).
    Builds a DataFrame with week, home_team, away_team and placeholder model columns.
    """

    rapidapi_key = get_secret("RAPIDAPI_KEY", None)
    rapidapi_host = get_secret("RAPIDAPI_HOST", RAPIDAPI_HOST_DEFAULT)

    if rapidapi_key is None:
        st.error("❌ RAPIDAPI_KEY is not set in Streamlit secrets.")
        return pd.DataFrame()

    url = "https://api-american-football.p.rapidapi.com/games"
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": rapidapi_host,
    }
    params = {
        "league": NFL_LEAGUE_ID,
        "season": str(season),
        # Depending on the API you might add filters like 'status=finished'
    }

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    games = data.get("response", data)

    rows = []

    for g in games:
        # ---- WEEK ----
        week_raw = g.get("week") or g.get("round") or None
        week_val = None

        if isinstance(week_raw, dict):
            # Some APIs use {"name": "Week 1", "id": 1}
            wk = week_raw.get("name") or week_raw.get("id") or week_raw.get("number")
        else:
            wk = week_raw

        if isinstance(wk, int):
            week_val = wk
        elif isinstance(wk, str):
            m = re.search(r"\d+", wk)
            if m:
                week_val = int(m.group(0))

        # ---- TEAMS ----
        teams = g.get("teams") or {}
        home_info = teams.get("home") or {}
        away_info = teams.get("away") or {}

        home_team = home_info.get("code") or home_info.get("name")
        away_team = away_info.get("code") or away_info.get("name")

        if home_team is None or away_team is None:
            continue  # skip bad records

        # ---- PLACEHOLDER MODEL VALUES ----
        model_home_win_prob = 0.50
        model_away_win_prob = 0.50
        edge_spread = 0.0
        edge_moneyline = 0.0

        rows.append(
            {
                "week": week_val,
                "home_team": home_team,
                "away_team": away_team,
                "model_home_win_prob": model_home_win_prob,
                "model_away_win_prob": model_away_win_prob,
                "edge_spread": edge_spread,
                "edge_moneyline": edge_moneyline,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        st.error("❌ No games returned from RapidAPI. Check league/season or your subscription.")
        return df

    # Clean week column a bit
    if "week" in df.columns:
        try:
            df["week"] = df["week"].fillna(0).astype(int)
        except Exception:
            pass

    # Sort for nicer display
    df = df.sort_values(["week", "home_team", "away_team"])

    return df


# ------------------------------------
# GET MODEL EDGES (SCHEDULE + MODEL)
# ------------------------------------
@st.cache_data(ttl=60 * 60)
def get_model_edges() -> pd.DataFrame:
    """
    Try to load a local CSV called model_edges_2025.csv.
    If not found, fallback to fetching schedule from RapidAPI and using placeholder model values.
    """

    # 1) Try reading from CSV (if you later want to upload your own model file)
    try:
        df = pd.read_csv("model_edges_2025.csv")
        st.success("✅ Loaded model_edges_2025.csv from repo.")
        return df
    except FileNotFoundError:
        st.info("ℹ️ No model_edges_2025.csv found. Fetching schedule from RapidAPI instead...")
    except Exception as e:
        st.warning(f"⚠️ Could not read model_edges_2025.csv ({e}). Using RapidAPI schedule instead...")

    # 2) Fallback → build from RapidAPI schedule
    df = fetch_schedule_from_rapidapi(SEASON)
    if df.empty:
        # If even RapidAPI failed, use tiny dummy data so app still runs
        st.warning("⚠️ Using dummy data because RapidAPI schedule failed.")
        df = pd.DataFrame(
            [
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
            ]
        )
    return df


edges_df = get_model_edges()

st.write("### DEBUG – Columns in edges_df")
st.write(list(edges_df.columns))


# ------------------------------------
# WEEK SELECTOR + MODEL TABLE
# ------------------------------------
if "week" in edges_df.columns:
    weeks = sorted(edges_df["week"].dropna().astype(int).unique().tolist())
    if len(weeks) == 0:
        selected_week = None
        week_df = edges_df.copy()
        st.warning("⚠️ No valid week numbers found.")
    else:
        default_idx = 0
        selected_week = st.sidebar.selectbox("Select NFL Week", weeks, index=default_idx)
        week_df = edges_df[edges_df["week"] == selected_week].copy()
else:
    selected_week = None
    week_df = edges_df.copy()
    st.warning("⚠️ No 'week' column found; showing all games.")

st.subheader(f"📊 Model Edges – Week {selected_week if selected_week is not None else 'ALL'}")

model_cols = ["home_team", "away_team", "model_home_win_prob", "model_away_win_prob"]
if "edge_spread" in week_df.columns:
    model_cols.append("edge_spread")
if "edge_moneyline" in week_df.columns:
    model_cols.append("edge_moneyline")

st.dataframe(week_df[model_cols], use_container_width=True)


# ------------------------------------
# FETCH LIVE ODDS FROM THEODDSAPI
# ------------------------------------
@st.cache_data(ttl=60 * 5)
def fetch_live_odds():
    """
    Fetch live NFL odds from TheOddsAPI.
    Requires ODDS_API_KEY in Streamlit secrets.
    """

    odds_key = get_secret("ODDS_API_KEY", None)
    if odds_key is None:
        st.error("❌ ODDS_API_KEY is not set in Streamlit secrets.")
        return pd.DataFrame()

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

        bm = bookmakers[0]  # first sportsbook

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
st.subheader("💰 Live Odds + Model Edges (This Week)")

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

        # Optional: sort by biggest spread edge if available
        if "edge_spread" in merged.columns:
            merged = merged.sort_values("edge_spread", ascending=False)

        st.dataframe(merged, use_container_width=True)

except Exception as e:
    st.error(f"❌ Could not fetch or merge live odds: {e}")


# ------------------------------------
# HEAD-TO-HEAD TOOL (MODEL ONLY)
# ------------------------------------
st.markdown("---")
st.subheader("🤝 Head-to-Head Matchup (Model Only)")

if not edges_df.empty:
    teams = sorted(pd.unique(edges_df[["home_team", "away_team"]].values.ravel()))
    col1, col2 = st.columns(2)
    with col1:
        home_sel = st.selectbox("Home team", teams, index=0)
    with col2:
        away_sel = st.selectbox("Away team", teams, index=1 if len(teams) > 1 else 0)

    if st.button("Predict matchup"):
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
                # Flip if stored reversed
                row = edges_df[mask2].iloc[0]
                home_prob = row["model_away_win_prob"]
                away_prob = row["model_home_win_prob"]
            else:
                st.error("This matchup is not in your data yet.")
                home_prob = away_prob = None

            if home_prob is not None:
                st.write(f"**Model win probability {home_sel} (home): {home_prob:.1%}**")
                st.write(f"**Model win probability {away_sel} (away): {away_prob:.1%}**")
else:
    st.info("No model data available yet.")
