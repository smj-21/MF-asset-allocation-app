"""
TOPSIS multi-criteria ranking, per risk-profile criteria weights, and the
downloadable Excel methodology report.

Uses vector normalization (value / sqrt(sum of squares of column)) per the
original TOPSIS paper -- more robust to outliers than linear-max
normalization, which matters given fund return data can have extreme values.
"""

import io

import numpy as np
import pandas as pd

CATEGORIES = ["largecap", "midcap", "flexicap", "smallcap", "gold"]

GENERAL_METHODOLOGY = """
**TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution) ranks each
fund by how close it is to an ideal *best* fund and how far it is from an ideal *worst*
fund, across all five criteria at once — rather than relying on any single number like
just returns or just risk.

**How the ranking is built, step by step:**

1. **Normalize** each criterion (1Y Return, 3Y Return, Volatility, Sharpe Ratio, Expense
   Ratio) using *vector normalization* — each fund's raw value is divided by the square
   root of the sum of squares of that entire column. This method, from the original 1981
   Hwang & Yoon TOPSIS paper, is more robust to outlier funds with unusually high or low
   values than simple min-max scaling.
2. **Weight** the normalized values using the risk-profile-specific weights shown below —
   criteria that matter more to your profile count for more in the final score.
3. **Identify the ideal best and ideal worst** for each criterion. For *benefit* criteria
   (1Y/3Y Return, Sharpe Ratio — higher is better), the ideal best is the maximum weighted
   value and the ideal worst is the minimum. For *cost* criteria (Volatility, Expense Ratio
   — lower is better), this is reversed.
4. **Measure distance**: for every fund, calculate its Euclidean distance to the ideal best
   and to the ideal worst, across all five weighted criteria simultaneously.
5. **Compute the closeness score**: distance-to-worst ÷ (distance-to-best +
   distance-to-worst). This produces a single score between 0 and 1 — a fund that is
   simultaneously close to the ideal and far from the worst scores near 1.
6. **Rank** funds within each category by this closeness score, descending. The top 5 are
   shown here, with your category's rupee amount split across them in proportion to their
   scores.

**Why AUM isn't part of this**: AUM (fund size) is already used earlier to shortlist the
initial 10 funds per category — including it again here would double-count the same signal.
"""

TOPSIS_WEIGHTS_BY_PROFILE = {
    "risk_averse": {
        "return_1y_calc": 0.10, "return_3y_calc": 0.20, "sharpe_ratio": 0.35,
        "volatility": 0.25, "expense_ratio": 0.10,
    },
    "conservative": {
        "return_1y_calc": 0.15, "return_3y_calc": 0.25, "sharpe_ratio": 0.30,
        "volatility": 0.20, "expense_ratio": 0.10,
    },
    "balanced": {
        "return_1y_calc": 0.20, "return_3y_calc": 0.30, "sharpe_ratio": 0.25,
        "volatility": 0.15, "expense_ratio": 0.10,
    },
    "growth": {
        "return_1y_calc": 0.25, "return_3y_calc": 0.30, "sharpe_ratio": 0.20,
        "volatility": 0.15, "expense_ratio": 0.10,
    },
    "aggressive": {
        "return_1y_calc": 0.30, "return_3y_calc": 0.30, "sharpe_ratio": 0.15,
        "volatility": 0.15, "expense_ratio": 0.10,
    },
}

CRITERIA_TYPES = {
    "return_1y_calc": "benefit", "return_3y_calc": "benefit", "sharpe_ratio": "benefit",
    "volatility": "cost", "expense_ratio": "cost",
}

PROFILE_REASONING = {
    "risk_averse": (
        "Risk Averse investors prioritize capital stability over chasing returns. "
        "Sharpe Ratio (35%) and Volatility (25%) together make up 60% of the score, "
        "rewarding funds that deliver steady, risk-adjusted performance rather than "
        "high but unstable returns. Raw 1-Year Return is weighted lowest (10%) since "
        "a single strong year is not a reliable signal for a conservative investor."
    ),
    "conservative": (
        "Conservative investors still favor stability but can tolerate slightly more "
        "return-seeking behavior than Risk Averse investors. Sharpe Ratio (30%) remains "
        "the top-weighted criterion, with Volatility (20%) and 3-Year Return (25%) "
        "balancing consistency against long-term growth."
    ),
    "balanced": (
        "Balanced investors weigh growth and stability roughly equally. Weights are "
        "spread more evenly across all five criteria, with 3-Year Return (30%) and "
        "Sharpe Ratio (25%) as the two largest factors, reflecting a preference for "
        "consistent long-term performance without ignoring downside risk."
    ),
    "growth": (
        "Growth investors are willing to accept more volatility in exchange for higher "
        "returns. 1-Year Return (25%) and 3-Year Return (30%) together make up 55% of "
        "the score, while Sharpe Ratio's weight is reduced (20%) relative to more "
        "conservative profiles, since short-term risk-adjustment matters less here."
    ),
    "aggressive": (
        "Aggressive investors prioritize maximizing returns and are comfortable with "
        "significant volatility. 1-Year and 3-Year Return together make up 60% of the "
        "score -- the highest return-weighting of any profile -- while Sharpe Ratio's "
        "influence is reduced further (15%), reflecting lower sensitivity to short-term "
        "risk-adjusted performance in pursuit of growth."
    ),
}


def get_topsis_criteria(risk_profile):
    weights = TOPSIS_WEIGHTS_BY_PROFILE[risk_profile]
    return {crit: {"weight": w, "type": CRITERIA_TYPES[crit]} for crit, w in weights.items()}


def run_topsis(category_df, criteria):
    """Vector-normalized TOPSIS ranking. Returns category_df sorted by rank,
    with 'topsis_score' and 'topsis_rank' columns added."""
    df = category_df.copy().reset_index(drop=True)
    criterion_names = list(criteria.keys())

    for crit in criterion_names:
        if df[crit].isna().any():
            fill_value = df[crit].mean()
            df[crit] = df[crit].fillna(fill_value)

    norm_df = pd.DataFrame(index=df.index)
    for crit in criterion_names:
        denom = np.sqrt((df[crit] ** 2).sum())
        norm_df[crit] = df[crit] / denom

    weighted_df = pd.DataFrame(index=df.index)
    for crit in criterion_names:
        weighted_df[crit] = norm_df[crit] * criteria[crit]["weight"]

    ideal_best = {}
    ideal_worst = {}
    for crit in criterion_names:
        if criteria[crit]["type"] == "benefit":
            ideal_best[crit] = weighted_df[crit].max()
            ideal_worst[crit] = weighted_df[crit].min()
        else:
            ideal_best[crit] = weighted_df[crit].min()
            ideal_worst[crit] = weighted_df[crit].max()

    dist_best = np.sqrt(sum((weighted_df[c] - ideal_best[c]) ** 2 for c in criterion_names))
    dist_worst = np.sqrt(sum((weighted_df[c] - ideal_worst[c]) ** 2 for c in criterion_names))

    closeness = dist_worst / (dist_best + dist_worst)

    df["topsis_score"] = closeness
    df["topsis_rank"] = df["topsis_score"].rank(ascending=False).astype(int)

    return df.sort_values("topsis_rank").reset_index(drop=True)


def build_topsis_excel_bytes(topsis_results, selected_profile, selected_criteria):
    """Builds the same TOPSIS methodology workbook as the Colab version, but
    returns it as in-memory bytes so Streamlit's download_button can serve it
    without writing to disk."""
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_rows = []
        for category, top5 in topsis_results.items():
            for _, row in top5.iterrows():
                summary_rows.append({
                    "Category": category,
                    "Fund Name": row["fund_name"],
                    "1Y Return (%)": row["return_1y_calc"],
                    "3Y Return (%)": row["return_3y_calc"],
                    "Volatility (%)": row["volatility"],
                    "Sharpe Ratio": row["sharpe_ratio"],
                    "Expense Ratio (%)": row["expense_ratio"],
                    "TOPSIS Score": round(row["topsis_score"], 4),
                    "Rank": row["topsis_rank"],
                    "Amount (₹)": round(row["fund_amount"], 2),
                })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        for category, top5 in topsis_results.items():
            sheet_name = f"TOPSIS_{category}"[:31]
            top5.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)

            ws = writer.sheets[sheet_name]
            ws["A1"] = f"TOPSIS Ranking — {category.upper()} — Risk Profile: {selected_profile.upper()}"

            footnote_row = top5.shape[0] + 5
            ws.cell(row=footnote_row, column=1, value="Weights used for this risk profile:")
            for i, (crit, details) in enumerate(selected_criteria.items()):
                ws.cell(
                    row=footnote_row + 1 + i, column=1,
                    value=f"  {crit}: {details['weight']*100:.0f}% ({details['type']})",
                )

            note_row = footnote_row + len(selected_criteria) + 2
            ws.cell(row=note_row, column=1, value="Why these weights:")
            ws.cell(row=note_row + 1, column=1, value=PROFILE_REASONING[selected_profile])

    buffer.seek(0)
    return buffer.getvalue()
