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

with st.expander("📚 Understanding the parameters to judge MF"):
    st.markdown(
        """
### Understanding the Parameters to Judge Mutual Funds

Every fund shown in this tool is described by six numbers. Here's what each one actually
means, how it's calculated, what range is considered healthy, and how to use it when
judging a fund yourself — even outside this tool.

---

**1. AUM — Assets Under Management (₹ Crore)**
- **What it is**: the total market value of all the money a fund currently manages —
  every investor's contribution combined.
- **Formula**: sum of the current market value of all the fund's holdings + any cash it holds.
- **Why it matters**: a reasonable AUM signals investor trust and gives the fund enough
  scale to operate efficiently. But size cuts both ways depending on category.
- **What's a healthy range**:
  - Largecap / Flexicap: size matters less — even ₹50,000+ Cr funds can move in and out of large stocks easily.
  - Midcap / Smallcap: very large AUM (₹30,000+ Cr in Smallcap) can actually hurt performance — the fund manager struggles to buy/sell meaningfully-sized stakes in smaller companies without moving the stock price.
  - As a floor: generally avoid funds below ₹500–1,000 Cr AUM regardless of category — too small can mean higher costs per investor and, rarely, a risk of the fund being shut down or merged.
- **Used in this tool**: only for the initial top-10 shortlist per category — it's deliberately left out of the TOPSIS ranking itself to avoid double-counting the same signal twice.

---

**2. 1-Year Return (%)**
- **What it is**: how much the fund's NAV (Net Asset Value, i.e. its price per unit) has
  grown over the trailing 12 months.
- **Formula**: `(NAV today ÷ NAV 1 year ago − 1) × 100`
- **Why it matters**: shows recent momentum — useful, but noisy. A single strong or weak
  year is often driven by short-term market cycles, not fund manager skill.
- **What's a healthy range**: highly dependent on the market that year — equity categories
  can range anywhere from -10% to +40% in a single year. The number itself matters less
  than how it compares to the category average and its benchmark index over the same period.
- **Beginner tip**: don't chase the fund with the highest 1-year return alone — that's
  "recency bias." A fund at the top of the 1-year list is often near the bottom the
  following year.

---

**3. 3-Year Return / CAGR (%)**
- **What it is**: the fund's *annualized* growth rate over the last 3 years — smooths out
  single-year noise and reflects performance across more of a market cycle.
- **Formula**: `((NAV today ÷ NAV 3 years ago) ^ (1/3) − 1) × 100`
- **Why it matters**: consistency over multiple years is a much stronger signal of
  genuine fund quality than any single year's number.
- **What's a healthy range**: Indian equity markets have historically compounded at
  roughly 10–15% CAGR over long periods — a fund consistently near or above its
  category average over 3 years is a good sign.
- **Beginner tip**: always look at 3-year (or longer) returns alongside 1-year — a fund
  that's merely "average" over 3 years but had one spectacular recent year is riskier
  than it looks.

---

**4. Volatility / Standard Deviation (%)**
- **What it is**: how much the fund's returns swing above and below its own average —
  a pure measure of risk, independent of whether the returns were good or bad.
- **Formula**: standard deviation of periodic (daily/weekly) returns, annualized by
  multiplying by the square root of the number of periods in a year (√252 for daily, √52
  for weekly).
- **Why it matters**: higher volatility means bigger ups *and* downs — more suited to
  investors with a longer time horizon who can stomach short-term drops.
- **What's a healthy range** (broad guide, varies by market conditions):
  - Largecap: ~12–16%
  - Midcap: ~16–20%
  - Smallcap: ~20–26%
  - Gold: ~12–15%
  - Bank FD: ~0% (fixed, guaranteed — that's the whole point of it in this tool's allocation)
- **Beginner tip**: lower volatility isn't automatically "better" — it depends on your
  risk profile. A Risk Averse investor should prefer the lower end of a category's
  volatility range; an Aggressive investor may accept the higher end for higher
  return potential.

---

**5. Sharpe Ratio**
- **What it is**: possibly the single most useful number here — it measures *return per
  unit of risk taken*, not just raw return.
- **Formula**: `(Fund Return − Risk-Free Rate) ÷ Fund's Standard Deviation`
  (this tool uses the prevailing Bank FD rate, 6.5%, as the risk-free rate).
- **Why it matters**: a fund with a high raw return but a low Sharpe ratio is likely
  taking on excessive risk to get there — the return isn't "efficient." A fund with a
  slightly lower return but a higher Sharpe ratio is often the smarter pick.
- **What's a healthy range**:
  - Below 0: poor — the fund didn't even beat a risk-free FD after accounting for its risk
  - 0 to 1: acceptable, fairly typical
  - 1 to 2: good — strong risk-adjusted performance
  - Above 2: excellent — rare, and worth checking it's sustained, not a one-off
- **Beginner tip**: when comparing two similar funds, all else equal, prefer the one with
  the higher Sharpe ratio over the one with the higher raw return.

---

**6. Expense Ratio (%)**
- **What it is**: the annual fee the fund house charges to manage your money — deducted
  automatically and gradually from the fund's NAV, so you never see it as a separate
  charge, but you pay it every single year regardless of how the fund performs.
- **Formula**: `(Total annual fund operating costs ÷ Total fund assets) × 100`
- **Why it matters**: this is the one factor entirely within a fund's control (unlike
  market returns) and it compounds against you every year — a 1% difference in expense
  ratio can cost you a meaningful chunk of your final corpus over 20–30 years.
- **What's a healthy range** (Direct plans, which is what this tool uses — Direct plans
  skip the distributor commission that Regular plans carry, so they're always cheaper):
  - Index funds: ~0.2–0.3%
  - Largecap / Flexicap active funds: ~0.5–1.0%
  - Midcap / Smallcap active funds: ~0.5–1.3%
  - Gold ETFs: ~0.4–0.65%
- **Beginner tip**: lower is generally better, all else equal — but don't pick a fund
  *purely* because it's cheap. A slightly pricier fund with meaningfully better
  risk-adjusted (Sharpe) performance is usually still the better choice.

---

**Why the tool weighs these together instead of picking just one**: no single number
tells the full story — a fund can look great on returns while quietly taking on huge
risk, or look "safe" on volatility while charging fees that erode your gains over time.
That's exactly why TOPSIS (see the methodology section after running the ranking) scores
funds across all five criteria at once, weighted according to how much each one should
matter for *your* risk profile.
        """
    )

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

    # ============================================================
    # STEP 5: Future value projection
    # ============================================================
    FD_PROJECTION_RATE = 0.075  # fixed 3-year FD rate assumption, per requirement

    st.divider()
    st.subheader("Step 5 — Future Value Projection")
    st.caption(
        "Each fund is projected forward using its own historical 3-Year CAGR — a smoother, "
        "more defensible assumption than a single noisy 1-year number. Bank FD is projected "
        "at a fixed 7.5% p.a. This is a projection based on historical performance, not a "
        "guarantee of future returns."
    )

    col_dur1, col_dur2 = st.columns(2)
    with col_dur1:
        duration_years_input = st.number_input("Years", min_value=0, value=5, step=1)
    with col_dur2:
        duration_months_input = st.number_input("Months", min_value=0, max_value=11, value=0, step=1)

    total_duration_years = duration_years_input + duration_months_input / 12

    if total_duration_years <= 0:
        st.warning("Enter a duration greater than 0 to see the projection.")
    else:
        fv_rows = []
        total_fv = 0.0

        for category in CATEGORIES:
            top5 = topsis_results[category]
            for _, row in top5.iterrows():
                invested = float(row["fund_amount"])
                annual_rate = float(row["return_3y_calc"]) / 100
                fv = invested * (1 + annual_rate) ** total_duration_years
                fv_rows.append({
                    "Category": category.capitalize(),
                    "Fund Name": row["fund_name"],
                    "Assumed Return (3Y CAGR %)": round(row["return_3y_calc"], 2),
                    "Amount Invested (₹)": invested,
                    "Future Value (₹)": fv,
                })
                total_fv += fv

        fd_amount = allocation["fd"] * amount
        fd_fv = fd_amount * (1 + FD_PROJECTION_RATE) ** total_duration_years
        fv_rows.append({
            "Category": "Bank FD",
            "Fund Name": "Bank Fixed Deposit",
            "Assumed Return (3Y CAGR %)": FD_PROJECTION_RATE * 100,
            "Amount Invested (₹)": fd_amount,
            "Future Value (₹)": fd_fv,
        })
        total_fv += fd_fv

        fv_df = pd.DataFrame(fv_rows)
        fv_display_df = fv_df.copy()
        fv_display_df["Amount Invested (₹)"] = fv_display_df["Amount Invested (₹)"].apply(format_inr)
        fv_display_df["Future Value (₹)"] = fv_display_df["Future Value (₹)"].apply(format_inr)
        st.dataframe(fv_display_df, hide_index=True, use_container_width=True)

        total_profit = total_fv - amount
        total_cagr = ((total_fv / amount) ** (1 / total_duration_years) - 1) * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Amount Invested", f"₹{format_inr(amount)}")
        m2.metric("Total Future Value", f"₹{format_inr(total_fv)}")
        m3.metric("Overall CAGR", f"{total_cagr:.2f}%")
        m4.metric(
            "Total Profit / Loss",
            f"₹{format_inr(total_profit)}",
            delta=f"{(total_profit / amount) * 100:.2f}%",
        )
