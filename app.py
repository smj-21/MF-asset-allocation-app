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
TOPSIS_TABLE_COLS = FUND_TABLE_COLS[:1] + [
    "return_1y_calc", "return_3y_calc", "volatility", "sharpe_ratio",
    "expense_ratio", "topsis_score", "topsis_rank", "fund_amount",
]

st.set_page_config(page_title="Guided Asset Allocation Tool", page_icon="📊", layout="wide")


def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.step = "input"


if "step" not in st.session_state:
    st.session_state.step = "input"

st.title("📊 Guided Asset Allocation Tool")
st.caption("Risk-profile-driven mutual fund allocation, shortlisting, and TOPSIS ranking")

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

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(alloc_df, hide_index=True, use_container_width=True)
    with col2:
        fig_pie = px.pie(alloc_df, names="Category", values="Amount (₹)", hole=0.4)
        fig_pie.update_traces(
            textinfo="percent+label",
            hovertemplate="%{label}<br>₹%{value:,.2f} (%{percent})<extra></extra>",
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
    st.dataframe(alloc_df, hide_index=True, use_container_width=True)

    for category in CATEGORIES:
        st.markdown(f"**{category.capitalize()}**")
        cat_df = final_df[final_df["category"] == category][FUND_TABLE_COLS]
        st.dataframe(cat_df, hide_index=True, use_container_width=True)

    st.button("🔄 Start Over", on_click=reset_app)

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

    # Stacked bar: one bar per category, split into 5 segments (one per
    # ranked fund), segment height = that fund's allocated rupee amount.
    fig_stack = go.Figure()
    for rank in range(1, 6):
        seg_amounts = []
        seg_fund_names = []
        for category in CATEGORIES:
            top5 = topsis_results[category]
            row = top5[top5["topsis_rank"] == rank]
            if not row.empty:
                seg_amounts.append(float(row["fund_amount"].iloc[0]))
                seg_fund_names.append(row["fund_name"].iloc[0])
            else:
                seg_amounts.append(0.0)
                seg_fund_names.append("")

        fig_stack.add_bar(
            x=[c.capitalize() for c in CATEGORIES],
            y=seg_amounts,
            name=f"Rank {rank}",
            text=seg_fund_names,
            hovertemplate="%{x}<br>%{text}<br>₹%{y:,.2f}<extra>Rank %{fullData.name}</extra>",
        )

    fig_stack.update_layout(
        barmode="stack",
        xaxis_title="Category",
        yaxis_title="Amount (₹)",
        legend_title="Rank",
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    for category in CATEGORIES:
        top5 = topsis_results[category]
        category_amount = allocation[category] * amount
        st.markdown(f"**{category.capitalize()}** — ₹{category_amount:,.2f}")
        st.dataframe(top5[TOPSIS_TABLE_COLS], hide_index=True, use_container_width=True)

    from engine.topsis import PROFILE_REASONING

    with st.expander("📖 View TOPSIS methodology"):
        st.write(PROFILE_REASONING[risk_profile])
        st.markdown("**Weights used for this risk profile:**")
        for crit, details in criteria.items():
            st.write(f"- {crit}: {details['weight']*100:.0f}% ({details['type']})")

    excel_bytes = build_topsis_excel_bytes(topsis_results, risk_profile, criteria)
    st.download_button(
        "📥 Download TOPSIS Excel Report",
        data=excel_bytes,
        file_name="topsis_model.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.button("🔄 Start Over", on_click=reset_app)
