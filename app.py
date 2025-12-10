import streamlit as st
import pandas as pd
import numpy as np
import nflreadpy as nfl
import requests

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

SEASON = 2025  # change this if you want a different season in the future


# ========= HELPER FUNCTIONS (BETTING MATH) =========

def american_to_prob(odds: float) -> float:
    """
    Convert American odds to implied probability.
    Example: -150 -> 0.60, +200 -> 0.333...
    """
    if odds is None:
        return None
    if pd.isna(odds):
        return None
    if odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return -odds / (-odds + 100.0)


def prob_and_odds_to_kelly_edge(p: float, odds: float):
    """
    Given model probability p and American odds, return:
      edge (p - implied_prob) and Kelly fraction.
    Kelly is based on moneyline (not spread).
    """
    implied = american_to_prob(odds)
    if implied is None:
        return None, None

    edge = p - implied  # positive = model > market

    # American -> decimal odds
    if odds > 0:
        decimal_odds = 1.0 + odds / 100.0
    else:
        decimal_odds = 1.0 + 100.0 / -odds

    b = decimal_odds - 1.0  # net profit per 1 stake if win

    # Kelly formula: f* = (b*p - (1-p)) / b
    kelly_fraction = (b * p - (1.0 - p)) / b
    if kelly_fraction < 0:
        kelly_fraction = 0.0

    return edge, kelly_fraction


# ========= DATA LOADERS =========

@st.cache_data(show_spinner=True)
def load_schedule_and_pbp(season: int):
    # ---- schedule ----
    sched = nfl.load_schedules([season]).to_pandas()

    # Keep regular season
    if "game_type" in sched.columns:
        sched_reg = sched[sched["game_type"] == "REG"].copy()
    elif "season_type" in sched.columns:
        sched_reg = sched[sched["season_type"] == "REG"].copy()
    else:
        if "week" in sched.columns:
            sched_reg = sched[sched["week"].between(1, 18)].copy()
        else:
            sched_reg = sched.copy()

    # Check needed columns
    for col in ["game_id", "week", "home_team", "away_team", "home_score", "away_score"]:
        if col not in sched_reg.columns:
            raise ValueError(f"Missing '{col}' in schedule data")

    # Played / future games
    games_played = sched_reg[
        sched_reg["home_score"].notna() & sched_reg["away_score"].notna()
    ].copy()
    games_future = sched_reg[
        sched_reg["home_score"].isna() | sched_reg["away_score"].isna()
    ].copy()

    # Base training table
    games = games_played[[
        "game_id", "week", "home_team", "away_team", "home_score", "away_score"
    ]].copy()
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    # ---- play-by-play ----
    pbp = nfl.load_pbp([season]).to_pandas()

    if "game_type" in pbp.columns:
        pbp_reg = pbp[pbp["game_type"] == "REG"].copy()
    elif "season_type" in pbp.columns:
        pbp_reg = pbp[pbp["season_type"] == "REG"].copy()
    else:
        if "week" in pbp.columns:
            pbp_reg = pbp[pbp["week"].between(1, 18)].copy()
        else:
            pbp_reg = pbp.copy()

    for col in ["game_id", "posteam", "defteam", "epa"]:
        if col not in pbp_reg.columns:
            raise ValueError(f"Missing '{col}' in PBP data")

    pbp_reg = pbp_reg[~pbp_reg["epa"].isna()].copy()

    pass_col = "pass" if "pass" in pbp_reg.columns else None
    rush_col = "rush" if "rush" in pbp_reg.columns else None
    if pass_col is None or rush_col is None:
        raise ValueError("Couldn't find 'pass' and 'rush' columns in PBP data")

    return sched_reg, games, games_future, pbp_reg, pass_col, rush_col


@st.cache_data(show_spinner=True)
def build_team_epa(pbp_reg: pd.DataFrame, pass_col: str, rush_col: str):
    # ---------- OFFENSE ----------
    pbp_off = pbp_reg[pbp_reg["posteam"].notna()].copy()
    pbp_off["is_pass"] = pbp_off[pass_col] == 1
    pbp_off["is_rush"] = pbp_off[rush_col] == 1

    pbp_off["pass_epa"] = np.where(pbp_off["is_pass"], pbp_off["epa"], np.nan)
    pbp_off["rush_epa"] = np.where(pbp_off["is_rush"], pbp_off["epa"], np.nan)

    off_summary = (
        pbp_off.groupby(["game_id", "posteam"], as_index=False)
        .agg(
            off_plays=("epa", "size"),
            off_epa=("epa", "mean"),
            pass_epa=("pass_epa", "mean"),
            rush_epa=("rush_epa", "mean"),
        )
        .rename(columns={"posteam": "team"})
    )

    # ---------- DEFENSE ----------
    pbp_def = pbp_reg[pbp_reg["defteam"].notna()].copy()
    pbp_def["def_pass_epa"] = np.where(pbp_def[pass_col] == 1, pbp_def["epa"], np.nan)
    pbp_def["def_rush_epa"] = np.where(pbp_def[rush_col] == 1, pbp_def["epa"], np.nan)

    def_summary = (
        pbp_def.groupby(["game_id", "defteam"], as_index=False)
        .agg(
            def_plays=("epa", "size"),
            def_epa=("epa", "mean"),
            def_pass_epa=("def_pass_epa", "mean"),
            def_rush_epa=("def_rush_epa", "mean"),
        )
        .rename(columns={"defteam": "team"})
    )

    team_game = off_summary.merge(def_summary, on=["game_id", "team"], how="inner")

    # Season-level EPA (used for model + rankings)
    team_season = (
        team_game
        .groupby("team", as_index=False)
        .agg(
            off_epa=("off_epa", "mean"),
            pass_epa=("pass_epa", "mean"),
            rush_epa=("rush_epa", "mean"),
            def_epa=("def_epa", "mean"),
            def_pass_epa=("def_pass_epa", "mean"),
            def_rush_epa=("def_rush_epa", "mean"),
        )
    )

    return team_game, team_season


@st.cache_data(show_spinner=True)
def train_model(games: pd.DataFrame, team_season: pd.DataFrame):
    # Season EPA for home / away
    team_season_home = team_season.rename(columns={"team": "home_team"})
    team_season_home = team_season_home.rename(
        columns={c: "home_" + c for c in team_season_home.columns if c != "home_team"}
    )

    team_season_away = team_season.rename(columns={"team": "away_team"})
    team_season_away = team_season_away.rename(
        columns={c: "away_" + c for c in team_season_away.columns if c != "away_team"}
    )

    games_model = games[["game_id", "week", "home_team", "away_team", "home_win"]].copy()

    model_df = (
        games_model
        .merge(team_season_home, on="home_team", how="left")
        .merge(team_season_away, on="away_team", how="left")
    )

    feature_cols = [
        "home_off_epa", "home_pass_epa", "home_rush_epa", "home_def_epa",
        "away_off_epa", "away_pass_epa", "away_rush_epa", "away_def_epa",
    ]

    model_data = model_df.dropna(subset=feature_cols + ["home_win"]).copy()
    X = model_data[feature_cols].values
    y = model_data["home_win"].values

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=5000))
    ])

    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]

    metrics = {
        "log_loss": log_loss(y, probs),
        "brier_score": brier_score_loss(y, probs),
        "n_games": len(model_data),
    }

    results = model_data[["week", "home_team", "away_team", "home_win"]].copy()
    results["home_win_prob"] = probs

    return model, feature_cols, results, metrics, model_df


# ========= LIVE ODDS FETCHER =========

@st.cache_data(show_spinner=True, ttl=300)
def fetch_live_odds():
    """
    Fetch live NFL odds (moneyline + spread) from an external API.
    Uses Streamlit secrets:
      - ODDS_API_KEY
      - ODDS_API_URL
      - ODDS_REGION
      - ODDS_MARKETS

    Returns a DataFrame with:
      game_id_key, home_team, away_team,
      home_ml, away_ml, home_spread_odds, away_spread_odds, spread_point
    """
    api_key = st.secrets["ODDS_API_KEY"]
    base_url = st.secrets["ODDS_API_URL"]
    region = st.secrets.get("ODDS_REGION", "us")
    markets = st.secrets.get("ODDS_MARKETS", "h2h,spreads")

    params = {
        "apiKey": api_key,
        "regions": region,
        "markets": markets,
        "oddsFormat": "american",
    }

    resp = requests.get(base_url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    rows = []

    for game in data:
        home = game["home_team"]
        away = game["away_team"]

        # We'll match on "Away @ Home"
        game_key = f"{away} @ {home}"

        home_ml = away_ml = None
        home_spread_odds = away_spread_odds = None
        spread_point = None

        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue

        # Take the first book (you can change to a specific book by name if you want)
        book = bookmakers[0]

        for market in book.get("markets", []):
            if market["key"] == "h2h":
                for out in market.get("outcomes", []):
                    if out["name"] == home:
                        home_ml = out["price"]
                    elif out["name"] == away:
                        away_ml = out["price"]

            elif market["key"] == "spreads":
                for out in market.get("outcomes", []):
                    if out["name"] == home:
                        home_spread_odds = out["price"]
                        spread_point = out["point"]
                    elif out["name"] == away:
                        away_spread_odds = out["price"]

        rows.append({
            "game_id_key": game_key,
            "home_team": home,
            "away_team": away,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread_odds": home_spread_odds,
            "away_spread_odds": away_spread_odds,
            "spread_point": spread_point,
        })

    return pd.DataFrame(rows)


# ========= MAIN APP =========

def main():
    st.title("NFL 2025 EPA Betting Model 🏈")
    st.write("Uses **2025-only EPA** to estimate win probabilities, then compares to **live moneyline & spread odds**.")

    # ---- load data & model ----
    with st.spinner("Loading data and training model..."):
        sched_reg, games, games_future, pbp_reg, pass_col, rush_col = load_schedule_and_pbp(SEASON)
        team_game, team_season = build_team_epa(pbp_reg, pass_col, rush_col)
        model, feature_cols, results, metrics, model_df = train_model(games, team_season)

    # ---- performance ----
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Games used", metrics["n_games"])
    col2.metric("Log loss", f"{metrics['log_loss']:.3f}")
    col3.metric("Brier score", f"{metrics['brier_score']:.3f}")

    st.markdown("---")

    # ---- team rankings ----
    st.subheader("Team Rankings (Season EPA)")
    tab1, tab2, tab3 = st.tabs(["Offense", "Defense", "Full EPA Table"])

    with tab1:
        off_rank = team_season.sort_values("off_epa", ascending=False)
        st.dataframe(off_rank.reset_index(drop=True))

    with tab2:
        def_rank = team_season.sort_values("def_epa", ascending=True)
        st.dataframe(def_rank.reset_index(drop=True))

    with tab3:
        st.dataframe(team_season)

    st.markdown("---")

    # ---- historical probabilities ----
    st.subheader("Historical Game Probabilities (Played Games)")
    week_filter = st.slider(
        "Filter by week",
        int(results["week"].min()),
        int(results["week"].max()),
        (int(results["week"].min()), int(results["week"].max())),
    )
    mask = (results["week"] >= week_filter[0]) & (results["week"] <= week_filter[1])
    st.dataframe(
        results[mask]
        .sort_values(["week", "home_win_prob"], ascending=[True, False])
        .reset_index(drop=True)
    )

    st.markdown("---")

    # ---- upcoming games (model only) ----
    st.subheader("Upcoming Games – Model Win Probabilities")

    preds_view = None

    if len(games_future) == 0:
        st.info("No future games left in the schedule for this season.")
    else:
        future_base = games_future[["game_id", "week", "home_team", "away_team"]].copy()

        # season EPA tables again
        team_season_home = team_season.rename(columns={"team": "home_team"})
        team_season_home = team_season_home.rename(
            columns={c: "home_" + c for c in team_season_home.columns if c != "home_team"}
        )
        team_season_away = team_season.rename(columns={"team": "away_team"})
        team_season_away = team_season_away.rename(
            columns={c: "away_" + c for c in team_season_away.columns if c != "away_team"}
        )

        future_df = (
            future_base
            .merge(team_season_home, on="home_team", how="left")
            .merge(team_season_away, on="away_team", how="left")
        )

        feature_cols_model = [
            "home_off_epa", "home_pass_epa", "home_rush_epa", "home_def_epa",
            "away_off_epa", "away_pass_epa", "away_rush_epa", "away_def_epa",
        ]

        pred_ready = future_df.dropna(subset=feature_cols_model, how="any").copy()
        if len(pred_ready) == 0:
            st.warning("Future games exist, but some teams do not have enough EPA data yet.")
        else:
            X_pred = pred_ready[feature_cols_model].values
            pred_ready["home_win_prob"] = model.predict_proba(X_pred)[:, 1]

            preds_sorted = pred_ready.sort_values("home_win_prob", ascending=False)
            preds_view = preds_sorted[["week", "home_team", "away_team", "home_win_prob"]].reset_index(drop=True)

            st.dataframe(preds_view)

    # ---- LIVE ODDS + EDGES FOR THIS WEEK ----
    st.markdown("---")
    st.subheader("Live Moneyline & Spread – Edges For This Week")

    if preds_view is None or len(preds_view) == 0:
        st.info("No upcoming games with model predictions yet.")
    else:
        try:
            # define "this week" as the next upcoming week
            this_week = int(preds_view["week"].min())
            st.write(f"Showing edges for **week {this_week}** upcoming games.")

            week_preds = preds_view[preds_view["week"] == this_week].copy()
            week_preds["game_id_key"] = week_preds["away_team"] + " @ " + week_preds["home_team"]

            odds_df = fetch_live_odds()
            if odds_df.empty:
                st.warning("No live odds returned from the odds API.")
            else:
                merged = week_preds.merge(odds_df, on="game_id_key", how="left")

                # implied probabilities from moneyline
                merged["home_implied"] = merged["home_ml"].apply(american_to_prob)
                merged["away_implied"] = merged["away_ml"].apply(american_to_prob)

                # edges
                merged["home_edge"] = merged["home_win_prob"] - merged["home_implied"]
                merged["away_win_prob"] = 1.0 - merged["home_win_prob"]
                merged["away_edge"] = merged["away_win_prob"] - merged["away_implied"]

                # best side
                merged["best_side"] = np.where(
                    merged["home_edge"] >= merged["away_edge"],
                    merged["home_team"],
                    merged["away_team"],
                )
                merged["best_side_is_home"] = merged["best_side"] == merged["home_team"]
                merged["best_side_prob"] = np.where(
                    merged["best_side_is_home"],
                    merged["home_win_prob"],
                    merged["away_win_prob"],
                )
                merged["best_side_odds"] = np.where(
                    merged["best_side_is_home"],
                    merged["home_ml"],
                    merged["away_ml"],
                )

                # best edge
                merged["best_side_edge"] = merged[["home_edge", "away_edge"]].max(axis=1)

                # convert to %
                merged["home_edge_pct"] = merged["home_edge"] * 100.0
                merged["away_edge_pct"] = merged["away_edge"] * 100.0
                merged["best_edge_pct"] = merged["best_side_edge"] * 100.0

                # Kelly suggestion (optional)
                kelly_vals = merged.apply(
                    lambda r: prob_and_odds_to_kelly_edge(r["best_side_prob"], r["best_side_odds"])
                    if pd.notna(r["best_side_odds"]) else (None, None),
                    axis=1,
                    result_type="expand",
                )
                merged["best_kelly_edge"], merged["best_kelly_fraction"] = kelly_vals[0], kelly_vals[1]

                display_cols = [
                    "week",
                    "home_team", "away_team",
                    "home_win_prob",
                    "home_ml", "home_implied", "home_edge_pct",
                    "away_ml", "away_implied", "away_edge_pct",
                    "spread_point", "home_spread_odds", "away_spread_odds",
                    "best_side", "best_side_prob", "best_side_odds", "best_edge_pct",
                    "best_kelly_fraction",
                ]

                merged_display = merged[display_cols].copy()

                EDGE_THRESHOLD = 5.0  # highlight edges >= 5 percentage points

                def highlight_big_edges(row):
                    color_row = [""] * len(row)
                    try:
                        edge = row["best_edge_pct"]
                        if pd.notna(edge) and edge >= EDGE_THRESHOLD:
                            color_row = ["background-color: rgba(0, 255, 0, 0.25)"] * len(row)
                    except Exception:
                        pass
                    return color_row

                styled = merged_display.style.apply(highlight_big_edges, axis=1)

                st.dataframe(styled, use_container_width=True)
                st.caption(
                    f"Rows highlighted when model edge on the best side is ≥ {EDGE_THRESHOLD:.1f}%."
                )

        except Exception as e:
            st.error(f"Could not fetch or merge live odds: {e}")

    st.markdown("---")

    # ---- simple matchup tool (no odds) ----
    st.subheader("Head-to-Head Matchup (Model Only)")

    teams_sorted = sorted(team_season["team"].unique())
    col_home2, col_away2 = st.columns(2)
    h2 = col_home2.selectbox("Home team", teams_sorted, key="home_model_only")
    a2 = col_away2.selectbox("Away team", teams_sorted, key="away_model_only")

    if st.button("Predict this matchup"):
        if h2 == a2:
            st.error("Home and away teams must be different.")
        else:
            th = team_season[team_season["team"] == h2].iloc[0]
            ta = team_season[team_season["team"] == a2].iloc[0]

            row2 = {
                "home_off_epa": th["off_epa"],
                "home_pass_epa": th["pass_epa"],
                "home_rush_epa": th["rush_epa"],
                "home_def_epa": th["def_epa"],
                "away_off_epa": ta["off_epa"],
                "away_pass_epa": ta["pass_epa"],
                "away_rush_epa": ta["rush_epa"],
                "away_def_epa": ta["def_epa"],
            }
            X_single2 = np.array([[row2[c] for c in [
                "home_off_epa", "home_pass_epa", "home_rush_epa", "home_def_epa",
                "away_off_epa", "away_pass_epa", "away_rush_epa", "away_def_epa",
            ]]])
            p_home = model.predict_proba(X_single2)[0, 1]
            p_away = 1.0 - p_home
            st.write(f"Home ({h2}) win probability: **{p_home:.3f}**")
            st.write(f"Away ({a2}) win probability: **{p_away:.3f}**")


if __name__ == "__main__":
    main()
