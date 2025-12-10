import streamlit as st
import pandas as pd
import numpy as np
import nflreadpy as nfl

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

SEASON = 2025  # you can make this a sidebar widget later if you want


@st.cache_data(show_spinner=True)
def load_schedule_and_pbp(season: int):
    # Load schedule
    sched = nfl.load_schedules([season]).to_pandas()

    # Try to keep only regular-season games
    if "game_type" in sched.columns:
        sched_reg = sched[sched["game_type"] == "REG"].copy()
    elif "season_type" in sched.columns:
        sched_reg = sched[sched["season_type"] == "REG"].copy()
    else:
        if "week" in sched.columns:
            sched_reg = sched[sched["week"].between(1, 18)].copy()
        else:
            sched_reg = sched.copy()

    # Played vs future
    for col in ["game_id", "week", "home_team", "away_team", "home_score", "away_score"]:
        if col not in sched_reg.columns:
            raise ValueError(f"Missing '{col}' in schedule data")

    games_played = sched_reg[
        sched_reg["home_score"].notna() & sched_reg["away_score"].notna()
    ].copy()
    games_future = sched_reg[
        sched_reg["home_score"].isna() | sched_reg["away_score"].isna()
    ].copy()

    # Build base games table for training
    games = games_played[[
        "game_id", "week", "home_team", "away_team", "home_score", "away_score"
    ]].copy()
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    # Load PBP
    pbp = nfl.load_pbp([season]).to_pandas()

    # Filter to regular season
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

    # Season-level EPA per team
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
    # Prepare season EPA tables
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


def main():
    st.title("NFL 2025 EPA Model (IG Edition) 🏈")
    st.write("Season-long EPA-based win probability model using 2025 data only.")

    with st.spinner("Loading data..."):
        sched_reg, games, games_future, pbp_reg, pass_col, rush_col = load_schedule_and_pbp(SEASON)
        team_game, team_season = build_team_epa(pbp_reg, pass_col, rush_col)
        model, feature_cols, results, metrics, model_df = train_model(games, team_season)

    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Games used", metrics["n_games"])
    col2.metric("Log loss", f"{metrics['log_loss']:.3f}")
    col3.metric("Brier score", f"{metrics['brier_score']:.3f}")

    st.markdown("---")

    st.subheader("Team Rankings (Season EPA)")
    tab1, tab2, tab3 = st.tabs(["Offense", "Defense", "Raw EPA table"])

    with tab1:
        off_rank = team_season.sort_values("off_epa", ascending=False)
        st.dataframe(off_rank.reset_index(drop=True))

    with tab2:
        def_rank = team_season.sort_values("def_epa", ascending=True)
        st.dataframe(def_rank.reset_index(drop=True))

    with tab3:
        st.dataframe(team_season)

    st.markdown("---")

    st.subheader("Historical Game Probabilities (Played Games)")
    week_filter = st.slider("Filter by week", int(results["week"].min()), int(results["week"].max()), (int(results["week"].min()), int(results["week"].max())))
    mask = (results["week"] >= week_filter[0]) & (results["week"] <= week_filter[1])
    st.dataframe(
        results[mask]
        .sort_values(["week", "home_win_prob"], ascending=[True, False])
        .reset_index(drop=True)
    )

    st.markdown("---")

    st.subheader("Upcoming Games – Model Predictions")

    if len(games_future) == 0:
        st.info("No future games left in the schedule for this season.")
    else:
        future_base = games_future[["game_id", "week", "home_team", "away_team"]].copy()

        # Season EPA tables again
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

        feature_cols = [
            "home_off_epa", "home_pass_epa", "home_rush_epa", "home_def_epa",
            "away_off_epa", "away_pass_epa", "away_rush_epa", "away_def_epa",
        ]

        pred_ready = future_df.dropna(subset=feature_cols, how="any").copy()

        if len(pred_ready) == 0:
            st.warning("Future games exist, but some teams do not have enough EPA data yet.")
        else:
            X_pred = pred_ready[feature_cols].values
            pred_ready["home_win_prob"] = model.predict_proba(X_pred)[:, 1]

            preds_sorted = pred_ready.sort_values("home_win_prob", ascending=False)
            st.dataframe(
                preds_sorted[["week", "home_team", "away_team", "home_win_prob"]]
                .reset_index(drop=True)
            )

    st.markdown("---")

    st.subheader("Head-to-Head Matchup Tool")

    teams_sorted = sorted(team_season["team"].unique())
    col_home, col_away = st.columns(2)
    home_team = col_home.selectbox("Home team", teams_sorted, key="home_team")
    away_team = col_away.selectbox("Away team", teams_sorted, key="away_team")

    if st.button("Predict this matchup"):
        if home_team == away_team:
            st.error("Home and away teams must be different.")
        else:
            # Build a fake game row from season EPA only
            th = team_season[team_season["team"] == home_team].iloc[0]
            ta = team_season[team_season["team"] == away_team].iloc[0]

            row = {
                "home_off_epa": th["off_epa"],
                "home_pass_epa": th["pass_epa"],
                "home_rush_epa": th["rush_epa"],
                "home_def_epa": th["def_epa"],
                "away_off_epa": ta["off_epa"],
                "away_pass_epa": ta["pass_epa"],
                "away_rush_epa": ta["rush_epa"],
                "away_def_epa": ta["def_epa"],
            }
            X_single = np.array([[row[c] for c in [
                "home_off_epa", "home_pass_epa", "home_rush_epa", "home_def_epa",
                "away_off_epa", "away_pass_epa", "away_rush_epa", "away_def_epa",
            ]]])
            prob = model.predict_proba(X_single)[0, 1]
            st.success(f"Model home win probability ({home_team} vs {away_team}): **{prob:.3f}**")

if __name__ == "__main__":
    main()
