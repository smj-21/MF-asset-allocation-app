"""
Allocation engine: inverse-volatility (risk-parity) weighting across the
five market categories, with the Bank FD share fixed per risk-profile
guardrail band and the remaining weight split across market categories by
inverse volatility.
"""

MARKET_CATEGORIES = ["largecap", "midcap", "flexicap", "smallcap", "gold"]

GUARDRAILS = {
    "risk_averse": {"fd": 0.60},
    "conservative": {"fd": 0.45},
    "balanced": {"fd": 0.25},
    "growth": {"fd": 0.20},
    "aggressive": {"fd": 0.10},
}

RISK_PROFILE_LABELS = {
    "risk_averse": "Risk Averse",
    "conservative": "Conservative",
    "balanced": "Balanced",
    "growth": "Growth",
    "aggressive": "Aggressive",
}


def get_allocation(risk_profile, volatility, guardrails=GUARDRAILS):
    """
    Args:
        risk_profile (str): one of the keys in GUARDRAILS
        volatility (pd.Series): annualized volatility, indexed by category
                                 (must include all of MARKET_CATEGORIES)
        guardrails (dict): FD weight per risk profile

    Returns:
        pd.Series: weights indexed by category, summing to 1.0
    """
    market_vol = volatility[MARKET_CATEGORIES]

    inv_vol = 1 / market_vol
    inv_vol_weights = inv_vol / inv_vol.sum()

    fd_weight = guardrails[risk_profile]["fd"]
    remaining_weight = 1 - fd_weight

    weights = inv_vol_weights * remaining_weight
    weights["fd"] = fd_weight

    return weights
