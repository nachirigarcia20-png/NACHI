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

    # Basic required columns
    for c in ["home_team", "away_team"]:
        if c not in df.columns:
            st.error(f"Missing required column in CSV: {c}")
            return None

    # Ensure week column
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

    # Moneyline edge if missing
    if "edge" not in df.columns and {"model_home_prob", "implied_home_prob"}.issubset(df.columns):
        df["edge"] = df["model_home_prob"] - df["implied_home_prob"]

    # Default bet columns
    for col in ["bet_units", "bet_units_spread", "bet_units_over", "bet_units_under"]:
        if col not in df.columns:
            df[col] = 0.0

    # Default spread / totals derived fields if missing
    if "edge_spread" not in df.columns:
        df["edge_spread"] = np.nan
    if "cover_prob_home" not in df.columns:
        df["cover_prob_home"] = np.nan
    if "prob_over" not in df.columns:
        df["prob_over"] = np.nan
    if "prob_under" not in df.columns:
        df["prob_under"] = np.nan
    if "edge_over" not in df.columns:
        df["edge_over"] = np.nan
    if "edge_under" not in df.columns:
        df["edge_under"] = np.nan

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

# ---------- HELPERS ----------
def prediction_from_margin(row):
    """Text prediction like 'KC by 3.4' based on model_margin."""
    margin = row.get("model_margin", np.nan)
    ht = row.get("home_team", "HOME")
    at = row.get("away_team", "AWAY")

    if pd.isna(margin):
        return ""
    if margin > 0.5:
        return f"{ht} by {margin:.1f}"
    elif margin < -0.5:
        return f"{at} by {abs(margin):.1f}"
    else:
        return "Coin flip"

def highlight_recommended(row):
    """Green row if any bet is recommended (ML, spread, or totals)."""
    ml = row.get("bet_units", 0.0) or 0.0
    sp = row.get("bet_units_spread", 0.0) or 0.0
    ou_over = row.get("bet_units_over", 0.0) or 0.0
    ou_under = row.get("bet_units_under", 0.0) or 0.0

    if (ml > 0) or (sp > 0) or (ou_over > 0) or (ou_under > 0):
        return ["background-color: #d1ffd1"] * len(row)
    else:
        return [""] * len(row)

def color_edge(val):
    """Green for positive edge, red for negative."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v > 0:
        return "color: green; font-weight: bold"
    elif v < 0:
        return "color: red"
    return ""

# ---------- HEADER ----------
st.title("🏈 IG Spin NFL Betting Model")
st.caption("Moneyline, spread and totals edges vs the market, with Kelly sizing.")

# ---------- TABS ----------
tab_overview, tab_ml, tab_spread, tab_totals = st.tabs(
    ["📊 Overview", "💰 Moneyline", "📉 Spread", "📈 Totals (O/U)"]
)

# ---------- OVERVIEW TAB ----------
with tab_overview:
    st.subheader("📊 Model Overview")

    if df_filtered.empty:
        st.info("No games match your current filters.")
    else:
        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            st.metric("Games in View", len(df_filtered))

        if "edge" in df_filtered.columns:
            top_ml_edge = df_filtered["edge"].max()
            with col_b:
                st.metric(
                    "Top Moneyline Edge",
                    f"{top_ml_edge * 100:.2f}%" if not np.isnan(top_ml_edge) else "—"
                )

        if "edge_spread" in df_filtered.columns:
            top_spread_edge = df_filtered["edge_spread"].max()
            with col_c:
                st.metric(
                    "Top Spread Edge",
                    f"{top_spread_edge * 100:.2f}%" if not np.isnan(top_spread_edge) else "—"
                )

        if "edge_over" in df_filtered.columns:
            top_ou_edge = max(
                df_filtered["edge_over"].max(),
                df_filtered["edge_under"].max()
            )
            with col_d:
                st.metric(
                    "Top Totals Edge",
                    f"{top_ou_edge * 100:.2f}%" if not np.isnan(top_ou_edge) else "—"
                )

        st.markdown("---")

        # Add prediction text
        if "model_margin" in df_filtered.columns:
            df_filtered = df_filtered.copy()
            df_filtered["prediction"] = df_filtered.apply(prediction_from_margin, axis=1)

        main_cols = [
            "week",
            "home_team",
            "away_team",
            "home_score",          # show scores so you can see past games
            "away_score",
            "home_market_ml",
            "home_market_spread",
            "market_total",
            "model_home_prob",
            "implied_home_prob",
            "edge",
            "model_margin",
            "cover_prob_home",
            "prob_over",
            "prob_under",
            "bet_units",
            "bet_units_spread",
            "bet_units_over",
            "bet_units_under",
            "prediction",
        ]
        main_cols = [c for c in main_cols if c in df_filtered.columns]

        table = df_filtered[main_cols].copy()
        styled = table.style

        # Row highlight
        styled = styled.apply(highlight_recommended, axis=1)

        # Per-column color for edges
        for col in ["edge", "edge_spread", "edge_over", "edge_under"]:
            if col in table.columns:
                styled = styled.applymap(color_edge, subset=[col])

        # Percent & numeric format
        fmt_dict = {}
        if "model_home_prob" in table.columns:
            fmt_dict["model_home_prob"] = "{:.1%}"
        if "implied_home_prob" in table.columns:
            fmt_dict["implied_home_prob"] = "{:.1%}"
        if "edge" in table.columns:
            fmt_dict["edge"] = "{:.2%}"
        if "cover_prob_home" in table.columns:
            fmt_dict["cover_prob_home"] = "{:.1%}"
        if "prob_over" in table.columns:
            fmt_dict["prob_over"] = "{:.1%}"
        if "prob_under" in table.columns:
            fmt_dict["prob_under"] = "{:.1%}"
        for c in ["bet_units", "bet_units_spread", "bet_units_over", "bet_units_under"]:
            if c in table.columns:
                fmt_dict[c] = "{:.2f}"

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

    if {"edge", "bet_units"}.issubset(bets_ml.columns):
        bets_ml = bets_ml[
            (bets_ml["edge"] * 100 >= min_edge_pct_ml) &
            (bets_ml["bet_units"] >= min_units_ml)
        ]

        if "edge" in bets_ml.columns:
            bets_ml = bets_ml.sort_values("edge", ascending=False)

        if bets_ml.empty:
            st.info("No moneyline bets meet your criteria.")
        else:
            if "model_margin" in bets_ml.columns:
                bets_ml = bets_ml.copy()
                bets_ml["prediction"] = bets_ml.apply(prediction_from_margin, axis=1)

            cols_ml = [
                "week",
                "home_team",
                "away_team",
                "home_market_ml",
                "model_home_prob",
                "implied_home_prob",
                "edge",
                "bet_units",
                "prediction",
            ]
            cols_ml = [c for c in cols_ml if c in bets_ml.columns]
            table_ml = bets_ml[cols_ml].copy()

            styled_ml = table_ml.style
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
            styled_ml = styled_ml.format(fmt_ml)

            st.dataframe(styled_ml, use_container_width=True)

            # Model-says list
            st.markdown("### 🧠 Model says (Moneyline)")
            for _, row in bets_ml.head(5).iterrows():
                ht = row["home_team"]
                at = row["away_team"]
                ml = row.get("home_market_ml", np.nan)
                p = row.get("model_home_prob", np.nan)
                edge_val = row.get("edge", np.nan)
                units = row.get("bet_units", 0.0)
                pred_text = prediction_from_margin(row)

                line_text = f"{ht} ML ({ml:+})" if not pd.isna(ml) else f"{ht} ML"
                prob_text = f"{p*100:.1f}%" if not pd.isna(p) else "—"
                edge_text = f"{edge_val*100:.1f}%" if not pd.isna(edge_val) else "—"
                units_text = f"{units:.2f}u"

                st.markdown(
                    f"- **{ht} vs {at}** — {line_text}  \n"
                    f"  Win prob: **{prob_text}**, edge: **{edge_text}**, stake: **{units_text}**  \n"
                    f"  Margin prediction: **{pred_text}**"
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
            if "model_margin" in bets_sp.columns:
                bets_sp = bets_sp.copy()
                bets_sp["prediction"] = bets_sp.apply(prediction_from_margin, axis=1)

            cols_sp = [
                "week",
                "home_team",
                "away_team",
                "home_market_spread",
                "cover_prob_home",
                "edge_spread",
                "bet_units_spread",
                "prediction",
            ]
            cols_sp = [c for c in cols_sp if c in bets_sp.columns]
            table_sp = bets_sp[cols_sp].copy()

            styled_sp = table_sp.style
            if "edge_spread" in table_sp.columns:
                styled_sp = styled_sp.applymap(color_edge, subset=["edge_spread"])

            fmt_sp = {}
            if "cover_prob_home" in table_sp.columns:
                fmt_sp["cover_prob_home"] = "{:.1%}"
            if "edge_spread" in table_sp.columns:
                fmt_sp["edge_spread"] = "{:.2%}"
            if "bet_units_spread" in table_sp.columns:
                fmt_sp["bet_units_spread"] = "{:.2f}"
            styled_sp = styled_sp.format(fmt_sp)

            st.dataframe(styled_sp, use_container_width=True)

            # Model-says list
            st.markdown("### 🧠 Model says (Spreads)")
            for _, row in bets_sp.head(5).iterrows():
                ht = row["home_team"]
                at = row["away_team"]
                spread = row.get("home_market_spread", np.nan)
                p_cover = row.get("cover_prob_home", np.nan)
                edge_sp = row.get("edge_spread", np.nan)
                units_sp = row.get("bet_units_spread", 0.0)
                pred_text = prediction_from_margin(row)

                line_text = f"{ht} {spread:+}" if not pd.isna(spread) else f"{ht} spread"
                prob_text = f"{p_cover*100:.1f}%" if not pd.isna(p_cover) else "—"
                edge_text = f"{edge_sp*100:.1f}%" if not pd.isna(edge_sp) else "—"
                units_text = f"{units_sp:.2f}u"

                st.markdown(
                    f"- **{ht} vs {at}** — {line_text}  \n"
                    f"  Cover prob: **{prob_text}**, edge: **{edge_text}**, stake: **{units_text}**  \n"
                    f"  Margin prediction: **{pred_text}**"
                )
    else:
        st.info("Spread columns not available in data.")

# ---------- TOTALS TAB ----------
with tab_totals:
    st.subheader("📈 Recommended Totals (Over/Under) Bets")

    bets_tot = df.copy()

    col7, col8, col9 = st.columns(3)
    with col7:
        min_edge_ou = st.slider("Min O/U edge (%)", 0.0, 20.0, 4.0, 0.5, key="ou_edge")
    with col8:
        min_units_ou = st.slider("Min O/U bet size (units)", 0.0, 10.0, 0.25, 0.25, key="ou_units")
    with col9:
        this_week_only_ou = st.checkbox("This week only", value=True, key="ou_week")

    needed_ou_cols = {"market_total", "model_total_points", "prob_over", "prob_under",
                      "edge_over", "edge_under", "bet_units_over", "bet_units_under"}

    if needed_ou_cols.issubset(bets_tot.columns):
        if this_week_only_ou and "week" in bets_tot.columns:
            current_week = bets_tot["week"].max()
            bets_tot = bets_tot[bets_tot["week"] == current_week]

        # keep rows where either over OR under meets criteria
        bets_tot = bets_tot[
            ((bets_tot["edge_over"] * 100 >= min_edge_ou) & (bets_tot["bet_units_over"] >= min_units_ou)) |
            ((bets_tot["edge_under"] * 100 >= min_edge_ou) & (bets_tot["bet_units_under"] >= min_units_ou))
        ]

        if bets_tot.empty:
            st.info("No totals bets meet your criteria.")
        else:
            cols_tot = [
                "week",
                "home_team",
                "away_team",
                "market_total",
                "model_total_points",
                "prob_over",
                "edge_over",
                "bet_units_over",
                "prob_under",
                "edge_under",
                "bet_units_under",
            ]
            cols_tot = [c for c in cols_tot if c in bets_tot.columns]
            table_tot = bets_tot[cols_tot].copy()

            # format
            if "prob_over" in table_tot.columns:
                table_tot["prob_over"] = (table_tot["prob_over"] * 100).round(1).astype(str) + "%"
            if "prob_under" in table_tot.columns:
                table_tot["prob_under"] = (table_tot["prob_under"] * 100).round(1).astype(str) + "%"
            for c in ["edge_over", "edge_under"]:
                if c in table_tot.columns:
                    table_tot[c] = (table_tot[c] * 100).round(2).astype(str) + "%"
            for c in ["bet_units_over", "bet_units_under"]:
                if c in table_tot.columns:
                    table_tot[c] = table_tot[c].round(2)

            st.dataframe(table_tot, use_container_width=True)

            # Model-says list
            st.markdown("### 🧠 Model says (Totals)")
            for _, row in bets_tot.head(5).iterrows():
                ht = row["home_team"]
                at = row["away_team"]
                line = row.get("market_total", np.nan)
                model_tot = row.get("model_total_points", np.nan)
                p_over = row.get("prob_over_raw", row.get("prob_over", np.nan))
                p_under = row.get("prob_under_raw", row.get("prob_under", np.nan))
                edge_over = row.get("edge_over_raw", row.get("edge_over", np.nan))
                edge_under = row.get("edge_under_raw", row.get("edge_under", np.nan))
                u_over = row.get("bet_units_over", 0.0)
                u_under = row.get("bet_units_under", 0.0)

                line_text = f"Total {line:.1f}" if not pd.isna(line) else "Total —"
                model_text = f"{model_tot:.1f}" if not pd.isna(model_tot) else "—"
                p_over_text = f"{p_over*100:.1f}%" if not pd.isna(p_over) else "—"
                p_under_text = f"{p_under*100:.1f}%" if not pd.isna(p_under) else "—"
                edge_over_text = f"{edge_over*100:.1f}%" if not pd.isna(edge_over) else "—"
                edge_under_text = f"{edge_under*100:.1f}%" if not pd.isna(edge_under) else "—"
                u_over_text = f"{u_over:.2f}u"
                u_under_text = f"{u_under:.2f}u"

                st.markdown(
                    f"- **{ht} vs {at}** — {line_text}  \n"
                    f"  Model total: **{model_text}**  \n"
                    f"  Over: prob **{p_over_text}**, edge **{edge_over_text}**, stake **{u_over_text}**  \n"
                    f"  Under: prob **{p_under_text}**, edge **{edge_under_text}**, stake **{u_under_text}**"
                )
    else:
        st.info("Totals columns not available. Make sure market_total, model_total_points, prob_over/under, edge_over/under, bet_units_over/under exist.")

st.caption("⚠️ Model outputs are for informational/educational purposes only.")
