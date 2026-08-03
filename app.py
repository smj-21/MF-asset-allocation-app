import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine.allocation import GUARDRAILS, RISK_PROFILE_LABELS, get_allocation
from engine.fund_data import fetch_all_fund_data
from engine.market_data import fetch_market_data
from engine.topsis import build_topsis_excel_bytes, get_topsis_criteria, run_topsis

CATEGORIES = ["largecap", "midcap", "flexicap", "smallcap", "gold"]
FUND_TABLE_COLS = [
    "fund_name", "aum_cr", "return_1y_calc", "return_3y_calc",
    "volatility", "sharpe_ratio", "expense_ratio",
]
TOPSIS_TABLE_COLS = [
    "topsis_rank", "fund_name", "fund_amount", "return_1y_calc", "return_3y_calc",
    "volatility", "sharpe_ratio", "expense_ratio", "topsis_score",
]
TOPSIS_COL_RENAME = {
    "topsis_rank": "Rank",
    "fund_name": "Fund Name",
    "return_1y_calc": "1Y Return (%)",
    "return_3y_calc": "3Y Return (%)",
    "volatility": "Volatility (%)",
    "sharpe_ratio": "Sharpe Ratio",
    "expense_ratio": "Expense Ratio (%)",
    "topsis_score": "TOPSIS Score",
    "fund_amount": "Amount (₹)",
}

# Category order here matches allocation.index (5 market categories + fd)
PIE_COLORS = ["#118AB2", "#06D6A0", "#FFD166", "#EF476F", "#7209B7", "#073B4C"]
# Rank 1 (best) = darkest/most prominent blue, fading lighter through Rank 5 —
# a single-hue gradient reads as more coherent than 5 unrelated colors.
RANK_COLORS = px.colors.sample_colorscale("Blues_r", [i / 4 for i in range(5)])


def _contrasting_text_color(rgb_string):
    """Pick black or white text depending on how light/dark the background is,
    so labels stay readable across the whole gradient (white text disappears
    on the lightest shade, black text disappears on the darkest)."""
    r, g, b = [float(v) for v in rgb_string.strip("rgb()").split(",")]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 150 else "white"


RANK_TEXT_COLORS = [_contrasting_text_color(c) for c in RANK_COLORS]


def format_inr(value, decimals=2):
    """Format a number using the Indian numbering system (last 3 digits,
    then groups of 2): 10000000 -> '1,00,00,000' instead of '10,000,000'."""
    is_negative = value < 0
    value = abs(value)
    num_str = f"{value:.{decimals}f}"
    integer_part, _, decimal_part = num_str.partition(".")

    if len(integer_part) > 3:
        last_three = integer_part[-3:]
        remaining = integer_part[:-3]
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        formatted_int = ",".join(groups) + "," + last_three
    else:
        formatted_int = integer_part

    result = f"{formatted_int}.{decimal_part}" if decimals > 0 else formatted_int
    return f"-{result}" if is_negative else result

st.set_page_config(page_title="Guided Asset Allocation Tool", page_icon="📊", layout="wide")


def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.step = "input"


if "step" not in st.session_state:
    st.session_state.step = "input"

st.title("📊 Guided Asset Allocation Tool")
st.caption("Risk-profile-driven shortlisting of mutual fund, FD and ETF allocation based on TOPSIS ranking")

with st.expander("👉 Click here to understand how this app works"):
    col_text, col_image = st.columns([2, 1])
    with col_text:
        st.markdown(
            """
### How This Tool Works

This tool builds a personalized investment portfolio — a mix of Mutual Funds, Fixed
Deposits, and Gold ETFs — tailored to your risk profile. It does this through four
guided steps, combining live market data with the same systematic techniques used in
institutional portfolio construction.

**Step 1 — Live Market Data**
- Pulls live weekly market data directly from Yahoo Finance (Largecap, Midcap, Flexicap, Smallcap, and Gold indices/ETFs)
- Pulls real mutual fund NAV history from AMFI's data via mfapi.in
- Nothing is static — every recommendation is grounded in current market behavior, refreshed regularly

**Step 2 — A Quantitative Allocation Model**
- Based on your selected risk profile, the tool decides how much goes into each asset category, in two layers:
- Fixed Deposit gets a set percentage tied directly to your risk profile — more conservative profiles hold more in FD, more aggressive profiles hold less
- The remaining amount is split across Largecap, Midcap, Flexicap, Smallcap, and Gold using an inverse-volatility model — calmer asset classes over the past year get a larger share, more volatile ones get a smaller share
- This is a systematic, risk-parity-style approach rather than a fixed or arbitrary split

**Step 3 — Multi-Criteria Fund Ranking (TOPSIS)**
- Within each category, the tool shortlists the top 10 funds by size (AUM)
- If you choose to proceed, it ranks these down to the top 5 using TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) — a multi-criteria decision-making method used in institutional finance, made interactive here
- Each fund is scored across five criteria: 1-year return, 3-year return, volatility, Sharpe ratio, and expense ratio — capturing performance and risk together, not returns alone
- The weight given to each criterion adjusts based on your risk profile — a Risk Averse investor's ranking leans on stability (Sharpe ratio, low volatility), while an Aggressive investor's ranking leans more on raw returns
- This means the fund recommendations themselves — not just the money split — reflect your risk appetite

**Step 4 — Personalized Results & Transparency**
- The final output brings everything together: category-wise allocation, ranked funds within each category, and the exact rupee amount to invest in each
- Results are visualized through interactive charts and tables
- A downloadable Excel file lays out the full TOPSIS calculation — scores, normalization, and a written explanation of why each criterion was weighted the way it was for your risk profile
- Nothing is a black box
            """
        )
    with col_image:
        st.image("assets/workflow.png", use_container_width=True)

st.markdown(
    """
    <style>
    div.stButton > button, div.stDownloadButton > button {
        font-size: 20px !important;
        font-weight: 600 !important;
        padding: 0.9em 2.2em !important;
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# STEP 1: Risk profile + investment amount
# ============================================================
if st.session_state.step == "input":
    st.subheader("Step 1 — Tell us about your investment")

    profile_label = st.selectbox("Select your risk profile", list(RISK_PROFILE_LABELS.values()))
    risk_profile = next(k for k, v in RISK_PROFILE_LABELS.items() if v == profile_label)

    amount = st.number_input(
        "Investment amount (₹)", min_value=1000.0, value=100000.0, step=1000.0, format="%.2f"
    )
    st.caption(f"You entered: ₹{format_inr(amount)}")

    st.caption(
        f"Bank FD guardrail for **{profile_label}**: "
        f"{GUARDRAILS[risk_profile]['fd']*100:.0f}% of your investment, fixed regardless of "
        f"market volatility. The rest is split across equity/gold by inverse volatility."
    )

    if st.button("Get my allocation →", type="primary"):
        st.session_state.risk_profile = risk_profile
        st.session_state.investment_amount = amount
        st.session_state.step = "allocation"
        st.rerun()

# ============================================================
# STEP 2: Fetch data, show allocation + top-10 shortlist, TOPSIS gate
# ============================================================
elif st.session_state.step == "allocation":
    risk_profile = st.session_state.risk_profile
    amount = st.session_state.investment_amount

    with st.spinner("Fetching live market data..."):
        returns, volatility = fetch_market_data()

    allocation = get_allocation(risk_profile, volatility)
    st.session_state.allocation = allocation

    st.subheader(f"Step 2 — Category Allocation ({RISK_PROFILE_LABELS[risk_profile]})")

    alloc_df = pd.DataFrame({
        "Category": [c.capitalize() for c in allocation.index],
        "Weight (%)": (allocation.values * 100).round(2),
        "Amount (₹)": (allocation.values * amount).round(2),
    })
    alloc_display_df = alloc_df.copy()
    alloc_display_df["Amount (₹)"] = alloc_display_df["Amount (₹)"].apply(format_inr)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(alloc_display_df, hide_index=True, use_container_width=True)
    with col2:
        pie_chart_df = alloc_df.copy()
        pie_chart_df["Category"] = pie_chart_df["Category"].str.upper()
        pie_chart_df["Amount Display"] = pie_chart_df["Amount (₹)"].apply(format_inr)
        fig_pie = px.pie(
            pie_chart_df, names="Category", values="Amount (₹)", hole=0.4,
            color_discrete_sequence=PIE_COLORS,
            custom_data=["Amount Display"],
        )
        fig_pie.update_traces(
            textinfo="percent+label",
            hovertemplate="%{label}<br>₹%{customdata[0]} (%{percent})<extra></extra>",
        )
        fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    if "final_df" not in st.session_state:
        progress_bar = st.progress(0.0, text="Starting fund data fetch...")

        def _update_progress(done, total):
            progress_bar.progress(done / total, text=f"Fetched {done}/{total} funds...")

        with st.spinner("Fetching mutual fund data from mfapi.in (~30-60s)..."):
            st.session_state.final_df = fetch_all_fund_data(_progress_callback=_update_progress)
        progress_bar.empty()

    final_df = st.session_state.final_df

    st.subheader("Top 10 Funds per Category")
    for category in CATEGORIES:
        with st.expander(category.capitalize(), expanded=False):
            cat_df = final_df[final_df["category"] == category][FUND_TABLE_COLS]
            st.dataframe(cat_df, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Step 3 — Want a ranked shortlist?")
    st.write(
        "We can run a TOPSIS multi-criteria ranking (1Y & 3Y return, volatility, "
        "Sharpe ratio, expense ratio) to narrow each category down to the top 5 funds."
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Yes, run TOPSIS ranking →", type="primary", use_container_width=True):
            st.session_state.step = "topsis"
            st.rerun()
    with c2:
        if st.button("No, this is enough", use_container_width=True):
            st.session_state.step = "done_no_topsis"
            st.rerun()

# ============================================================
# STEP 3a: Final output without TOPSIS
# ============================================================
elif st.session_state.step == "done_no_topsis":
    risk_profile = st.session_state.risk_profile
    amount = st.session_state.investment_amount
    allocation = st.session_state.allocation
    final_df = st.session_state.final_df

    st.success("Here's your final category allocation and fund shortlist.")

    alloc_df = pd.DataFrame({
        "Category": [c.capitalize() for c in allocation.index],
        "Weight (%)": (allocation.values * 100).round(2),
        "Amount (₹)": (allocation.values * amount).round(2),
    })
    alloc_df["Amount (₹)"] = alloc_df["Amount (₹)"].apply(format_inr)
    st.dataframe(alloc_df, hide_index=True, use_container_width=True)

    for category in CATEGORIES:
        st.markdown(f"**{category.capitalize()}**")
        cat_df = final_df[final_df["category"] == category][FUND_TABLE_COLS]
        st.dataframe(cat_df, hide_index=True, use_container_width=True)

    st.button("🔄 Start Over", on_click=reset_app, use_container_width=True, type="primary")

# ============================================================
# STEP 3b: TOPSIS ranking + Excel export
# ============================================================
elif st.session_state.step == "topsis":
    risk_profile = st.session_state.risk_profile
    amount = st.session_state.investment_amount
    allocation = st.session_state.allocation
    final_df = st.session_state.final_df

    if "topsis_results" not in st.session_state:
        criteria = get_topsis_criteria(risk_profile)
        topsis_results = {}

        for category in CATEGORIES:
            category_df = final_df[final_df["category"] == category]
            ranked = run_topsis(category_df, criteria)
            top5 = ranked.head(5).copy()

            category_amount = allocation[category] * amount
            score_sum = top5["topsis_score"].sum()
            top5["fund_amount"] = (top5["topsis_score"] / score_sum) * category_amount

            topsis_results[category] = top5

        st.session_state.topsis_results = topsis_results
        st.session_state.topsis_criteria = criteria

    topsis_results = st.session_state.topsis_results
    criteria = st.session_state.topsis_criteria

    st.subheader(f"Step 4 — Top 5 Ranked Funds ({RISK_PROFILE_LABELS[risk_profile]})")

    excel_bytes = build_topsis_excel_bytes(topsis_results, risk_profile, criteria)

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "📥 Download TOPSIS Excel Report",
            data=excel_bytes,
            file_name="topsis_model.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with col_b:
        st.button("🔄 Start Over", on_click=reset_app, use_container_width=True, type="primary")

    from engine.topsis import GENERAL_METHODOLOGY, PROFILE_REASONING

    with st.expander("📖 View TOPSIS methodology"):
        st.markdown(GENERAL_METHODOLOGY)
        st.markdown("---")
        st.markdown(f"**Why these weights for {RISK_PROFILE_LABELS[risk_profile]}:**")
        st.write(PROFILE_REASONING[risk_profile])
        st.markdown("**Weights used for this risk profile:**")
        for crit, details in criteria.items():
            st.write(f"- {crit}: {details['weight']*100:.0f}% ({details['type']})")

    # Treemap: Category > Fund, box size = allocated amount, colored by rank.
    # Handles very uneven category totals (e.g. Gold vs Largecap) without the
    # label-crowding a stacked bar gets when segments are small.
    # Stacked bar: one bar per category, split into 5 segments (one per
    # ranked fund), segment height = that fund's allocated rupee amount.
    fig_stack = go.Figure()
    for rank in range(1, 6):
        seg_amounts = []
        seg_labels = []
        for category in CATEGORIES:
            top5 = topsis_results[category]
            row = top5[top5["topsis_rank"] == rank]
            if not row.empty:
                fund_amount = float(row["fund_amount"].iloc[0])
                fund_name = row["fund_name"].iloc[0]
                seg_amounts.append(fund_amount)
                seg_labels.append(f"<b>{fund_name.upper()}</b><br><b>₹{format_inr(fund_amount, 0)}</b>")
            else:
                seg_amounts.append(0.0)
                seg_labels.append("")

        fig_stack.add_bar(
            x=[c.upper() for c in CATEGORIES],
            y=seg_amounts,
            name=f"Rank {rank}",
            text=seg_labels,
            textposition="inside",
            insidetextfont=dict(color=RANK_TEXT_COLORS[rank - 1], size=12),
            marker=dict(color=RANK_COLORS[rank - 1], line=dict(width=1, color="#0E1117")),
            hovertemplate="%{x}<br>%{text}<extra>RANK %{fullData.name}</extra>",
        )

    fig_stack.update_layout(
        barmode="stack",
        xaxis_title="CATEGORY",
        yaxis_title="AMOUNT (₹)",
        legend_title="RANK",
        margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    for category in CATEGORIES:
        top5 = topsis_results[category]
        category_amount = allocation[category] * amount
        st.markdown(f"**{category.capitalize()}** — ₹{format_inr(category_amount)}")
        display_df = top5[TOPSIS_TABLE_COLS].rename(columns=TOPSIS_COL_RENAME)
        display_df["Amount (₹)"] = display_df["Amount (₹)"].apply(format_inr)
        st.dataframe(display_df, hide_index=True, use_container_width=True)
