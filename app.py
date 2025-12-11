import streamlit as st
import pandas as pd
import numpy as np

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="IG Spin NFL Model",
    page_icon="🏈",
    layout="wide"
)

# ---------- DATA LOADER ----------
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

    # Make sure week exists
    if "week" not in df.columns:
        df["week"] = 0

    # Implied prob from ML if missing
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

    # Edge if missing
    if "edge" not in df.columns and {"model_home_prob", "implied_home_prob"}.issubset(df.columns):
        df["edge"] = df["model_home_prob"] - df["implied_home_prob"]

    # Default ML bet units
    if "bet_units" not in df.columns:
        df["bet_units"] = 0.0

    # Default spread fields
    if "edge_spread" not in df.columns:
        df["edge_spread"] = np.nan
    if "bet_units_spread" not in df.columns:
        df["bet_units_spread"] = 0.0

    # Predicted scores (only if we have total + margin model)
    if {"model_margin", "model_total_points"}.issubset(df.columns):
        df["pred_home_score"] = (df["model_total_points"] + df["model_margin"]) / 2.0
        df["pred_away_score"] = df["model_total_points"] - df["pred_home_score"]

    return df


df = load_data()
if df is None:
    st.stop()

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Filters")

weeks = sorted(df["week"].dropna().unique().tolist())
selected_week = st.sidebar.selectbox(
    "Week",
    options=["All"] + weeks,
    index=0
)

teams = sorted(set(df["home_team"]).union(set(df["away_team"])))
selected_teams = st.sidebar.multiselect(
    "Filter by team (home or away)",
    options=teams,
    default=[]
)

min_edge_sidebar = st.sidebar.slider(
    "Min ML edge (%) for highlights",
    0.0, 20.0, 4.0, 0.5
)

# Apply basic filters
df_filtered = df.copy()
if selected_week != "All":
    df_filtered = df_filtered[df_filtered["week"] == selected_week]

if selected_teams:
    df_filtered = df_filtered[
        df_filtered["home_team"].isin(selected_teams)
        | df_filtered["away_team"].isin(selected_teams)
    ]

# ---------- HEADER ----------
st.title("🏈 IG Spin NFL Betting Model")
st.caption("Model-driven edges vs the market — moneyline & spreads with Kelly sizing.")

# ---------- STYLE HELPERS ----------
def highlight_recommended(row):
    ml_rec = False
    sp_rec = False

    if "bet_units" in row and not pd.isna(row["bet_units"]):
        ml_rec = row["bet_units"] > 0

    if "bet_units_spread" in row and not pd.isna(row["bet_units_spread"]):
        sp_rec = row["bet_units_spread"] > 0

    if ml_rec or sp_rec:
        return ["background-color: #d1ffd1"] * len(row)  # light green row
    else:
        return [""] * len(row)

def color_edge(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v > 0:
        return "color: green; font-weight: bold"
    elif v < 0:
        return "color: red"
    else:
        return ""

# ---------- TABS ----------
tab_overview, tab_ml, tab_spread = st.tabs(
    ["📊 Overview", "💰 Moneyline Edges", "📉 Spread (ATS) Edges"]
)

# ---------- OVERVIEW TAB ----------
with tab_overview:
    st.subheader("📊 Model Overview")

    if df_filtered.empty:
        st.info("No games match your current filters.")
    else:
        # Summary metrics
        col_a, col_b, col_c = st.columns(3)

        # Number of games in view
        with col_a:
            st.metric(
                label="Games in View",
                value=len(df_filtered)
            )

        # Best ML edge
        if "edge" in df_filtered.columns:
            top_ml_edge = df_filtered["edge"].max()
            with col_b:
                st.metric(
                    label="Top Moneyline Edge",
                    value=f"{top_ml_edge * 100:.2f}%" if not np.isnan(top_ml_edge) else "—"
                )

        # Best spread edge
        if "edge_spread" in df_filtered.columns:
            top_spread_edge = df_filtered["edge_spread"].max()
            with col_c:
                st.metric(
                    label="Top Spread Edge",
                    value=f"{top_spread_edge * 100:.2f}%" if not np.isnan(top_spread_edge) else "—"
                )

        st.markdown("---")

        # Main games table
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
            "cover_prob_home",
            "bet_units",
            "bet_units_spread",
            "pred_home_score",
            "pred_away_score",
        ]
        main_cols = [c for c in main_cols if c in df_filtered.columns]

        table = df_filtered[main_cols].copy()

        styled = table.style

        # Row highlight for recommended bets
        styled = styled.apply(highlight_recommended, axis=1)

        # Green/red text for edges
        if "edge" in table.columns:
            styled = styled.applymap(color_edge, subset=["edge"])
        if "edge_spread" in table.columns and "edge_spread" in table.columns:
            styled = styled.applymap(color_edge, subset=["edge_spread"])

        # Percent and numeric formatting
        fmt_dict = {}
        if "model_home_prob" in table.columns:
            fmt_dict["model_home_prob"] = "{:.1%}"
        if "implied_home_prob" in table.columns:
            fmt_dict["implied_home_prob"] = "{:.1%}"
        if "edge" in table.columns:
            fmt_dict["edge"] = "{:.2%}"
        if "cover_prob_home" in table.columns:
            fmt_dict["cover_prob_home"] = "{:.1%}"
        if "bet_units" in table.columns:
            fmt_dict["bet_units"] = "{:.2f}"
        if "bet_units_spread" in table.columns:
            fmt_dict["bet_units_spread"] = "{:.2f}"
        if "pred_home_score" in table.columns:
            fmt_dict["pred_home_score"] = "{:.1f}"
        if "pred_away_score" in table.columns:
            fmt_dict["pred_away_score"] = "{:.1f}"

        styled = styled.format(fmt_dict)

        st.dataframe(styled, use_container_width=True)


# ---------- MONEYLINE TAB ----------
with tab_ml:
    st.subheader("💰 Recommended Moneyline Bets")

    bets_ml = df.copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        min_edge_pct_ml = st.slider("Min ML edge (%)", 0.0, 20.0, 4.0, 0.5, key="ml_edge")
    with col2:
        min_units_ml = st.slider("Min ML bet size (units)", 0.0, 10.0, 0.25, 0.25, key="ml_units")
    with col3:
        this_week_only_ml = st.checkbox("This week only", value=True, key="ml_week")

    if this_week_only_ml and "week" in bets_ml.columns:
        current_week = bets_ml["week"].max()
        bets_ml = bets_ml[bets_ml["week"] == current_week]

    if {"edge","bet_units"}.issubset(bets_ml.columns):
        bets_ml = bets_ml[
            (bets_ml["edge"] * 100 >= min_edge_pct_ml) &
            (bets_ml["bet_units"] >= min_units_ml)
        ]

        if "edge" in bets_ml.columns:
            bets_ml = bets_ml.sort_values("edge", ascending=False)

        if bets_ml.empty:
            st.info("No moneyline bets meet your criteria.")
        else:
            cols_ml = [
                "week",
                "home_team",
                "away_team",
                "home_market_ml",
                "model_home_prob",
                "implied_home_prob",
                "edge",
                "bet_units",
                "pred_home_score",
                "pred_away_score",
            ]
            cols_ml = [c for c in cols_ml if c in bets_ml.columns]
            table_ml = bets_ml[cols_ml].copy()

            styled_ml = table_ml.style

            # Green/red edges
            if "edge" in table_ml.columns:
                styled_ml = styled_ml.applymap(color_edge, subset=["edge"])

            fmt_ml = {}
            if "model_home_prob" in table_ml.columns:
                fmt_ml["model_home_prob"] = "{:.1%}"
            if "implied_home_prob" in table_ml.columns:
                fmt_ml["implied_home_prob"] = "{:.1%}"
            if "edge" in table_ml.columns:
                fmt_ml["edge"] = "{:.2%}"
            if "bet_units" in table_ml.columns:
                fmt_ml["bet_units"] = "{:.2f}"
            if "pred_home_score" in table_ml.columns:
                fmt_ml["pred_home_score"] = "{:.1f}"
            if "pred_away_score" in table_ml.columns:
                fmt_ml["pred_away_score"] = "{:.1f}"

            styled_ml = styled_ml.format(fmt_ml)

            st.dataframe(styled_ml, use_container_width=True)

            # 🔊 Model Says section
            st.markdown("### 🧠 Model says (moneyline)")
            max_cards = 5
            for _, row in bets_ml.head(max_cards).iterrows():
                ht = row["home_team"]
                at = row["away_team"]
                ml = row.get("home_market_ml", np.nan)
                p = row.get("model_home_prob", np.nan)
                edge_val = row.get("edge", np.nan)
                units = row.get("bet_units", 0.0)
                ph = row.get("pred_home_score", np.nan)
                pa = row.get("pred_away_score", np.nan)

                line_text = f"{ht} ML ({ml:+})" if not pd.isna(ml) else f"{ht} ML"
                prob_text = f"{p*100:.1f}%" if not pd.isna(p) else "—"
                edge_text = f"{edge_val*100:.1f}%" if not pd.isna(edge_val) else "—"
                units_text = f"{units:.2f}u"

                if not pd.isna(ph) and not pd.isna(pa):
                    score_text = f"Projected score: **{ht} {ph:.1f} – {at} {pa:.1f}**"
                else:
                    score_text = ""

                st.markdown(
                    f"- **{ht} vs {at}** — {line_text}  \n"
                    f"  Win prob: **{prob_text}**, edge: **{edge_text}**, stake: **{units_text}**  \n"
                    f"  {score_text}"
                )
    else:
        st.info("Moneyline columns not available in data.")


# ---------- SPREAD TAB ----------
with tab_spread:
    st.subheader("📉 Recommended Spread (ATS) Bets")

    bets_sp = df.copy()

    col4, col5, col6 = st.columns(3)
    with col4:
        min_edge_pct_sp = st.slider("Min spread edge (%)", 0.0, 20.0, 4.0, 0.5, key="sp_edge")
    with col5:
        min_units_sp = st.slider("Min spread bet size (units)", 0.0, 10.0, 0.25, 0.25, key="sp_units")
    with col6:
        this_week_only_sp = st.checkbox("This week only", value=True, key="sp_week")

    needed_sp_cols = {"edge_spread", "bet_units_spread", "home_market_spread", "cover_prob_home"}
    if needed_sp_cols.issubset(bets_sp.columns):
        if this_week_only_sp and "week" in bets_sp.columns:
            current_week = bets_sp["week"].max()
            bets_sp = bets_sp[bets_sp["week"] == current_week]

        bets_sp = bets_sp[
            (bets_sp["edge_spread"] * 100 >= min_edge_pct_sp) &
            (bets_sp["bet_units_spread"] >= min_units_sp)
        ]

        if "edge_spread" in bets_sp.columns:
            bets_sp = bets_sp.sort_values("edge_spread", ascending=False)

        if bets_sp.empty:
            st.info("No spread bets meet your criteria.")
        else:
            cols_sp = [
                "week",
                "home_team",
                "away_team",
                "home_market_spread",
                "cover_prob_home",
                "edge_spread",
                "bet_units_spread",
                "pred_home_score",
                "pred_away_score",
            ]
            cols_sp = [c for c in cols_sp if c in bets_sp.columns]
            table_sp = bets_sp[cols_sp].copy()

            styled_sp = table_sp.style

            # Green/red edge_spread
            if "edge_spread" in table_sp.columns:
                styled_sp = styled_sp.applymap(color_edge, subset=["edge_spread"])

            fmt_sp = {}
            if "cover_prob_home" in table_sp.columns:
                fmt_sp["cover_prob_home"] = "{:.1%}"
            if "edge_spread" in table_sp.columns:
                fmt_sp["edge_spread"] = "{:.2%}"
            if "bet_units_spread" in table_sp.columns:
                fmt_sp["bet_units_spread"] = "{:.2f}"
            if "pred_home_score" in table_sp.columns:
                fmt_sp["pred_home_score"] = "{:.1f}"
            if "pred_away_score" in table_sp.columns:
                fmt_sp["pred_away_score"] = "{:.1f}"

            styled_sp = styled_sp.format(fmt_sp)

            st.dataframe(styled_sp, use_container_width=True)

            # 🔊 Model Says section for spreads
            st.markdown("### 🧠 Model says (spreads)")
            max_cards_sp = 5
            for _, row in bets_sp.head(max_cards_sp).iterrows():
                ht = row["home_team"]
                at = row["away_team"]
                spread = row.get("home_market_spread", np.nan)
                p_cover = row.get("cover_prob_home", np.nan)
                edge_sp = row.get("edge_spread", np.nan)
                units_sp = row.get("bet_units_spread", 0.0)
                ph = row.get("pred_home_score", np.nan)
                pa = row.get("pred_away_score", np.nan)

                line_text = f"{ht} {spread:+}" if not pd.isna(spread) else f"{ht} spread"
                prob_text = f"{p_cover*100:.1f}%" if not pd.isna(p_cover) else "—"
                edge_text = f"{edge_sp*100:.1f}%" if not pd.isna(edge_sp) else "—"
                units_text = f"{units_sp:.2f}u"

                if not pd.isna(ph) and not pd.isna(pa):
                    score_text = f"Projected score: **{ht} {ph:.1f} – {at} {pa:.1f}**"
                else:
                    score_text = ""

                st.markdown(
                    f"- **{ht} vs {at}** — {line_text}  \n"
                    f"  Cover prob: **{prob_text}**, edge: **{edge_text}**, stake: **{units_text}**  \n"
                    f"  {score_text}"
                )
    else:
        st.info("Spread columns not available in data. Make sure cover_prob_home, edge_spread, bet_units_spread exist.")

st.caption("⚠️ Model outputs are for informational/educational purposes only.")
