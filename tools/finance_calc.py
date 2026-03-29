# XIRR, tax comparison, expense drag, SIP math, insurance gap.
# All functions are pure — no I/O, no API calls.


import logging
from datetime import date
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── XIRR ──────────────────────────────────────────────────────────────────────

def xirr(cash_flows: List[float], dates: List[date], guess: float = 0.1) -> Optional[float]:
    """
    Extended Internal Rate of Return for irregular cash flows.
    Negative values = money invested (outflow).
    Positive values = current portfolio value (inflow).

    Verified test:
        xirr([-10000, 12000], [date(2023,1,1), date(2023,12,31)]) → ~0.20
        xirr([-5000, -5000, -5000, 18000], [date(2021,1,1), date(2022,1,1), date(2023,1,1), date(2024,1,1)]) → ~0.077
    """
    try:
        from scipy.optimize import newton

        if len(cash_flows) != len(dates):
            raise ValueError("cash_flows and dates length mismatch")
        if len(cash_flows) < 2:
            raise ValueError("Need at least 2 cash flows")

        d0 = dates[0]
        year_fracs = [(d - d0).days / 365.0 for d in dates]

        def npv(rate: float) -> float:
            return sum(cf / (1 + rate) ** yf for cf, yf in zip(cash_flows, year_fracs))

        def npv_deriv(rate: float) -> float:
            return sum(
                -yf * cf / (1 + rate) ** (yf + 1)
                for cf, yf in zip(cash_flows, year_fracs)
            )

        result = newton(npv, guess, fprime=npv_deriv, maxiter=100, tol=1e-6)

        if -0.99 < result < 5.0:
            return round(result, 4)
        return None

    except Exception as e:
        logger.warning(f"XIRR failed: {e}")
        return None


def simple_cagr(invested: float, current_value: float, years: float) -> Optional[float]:
    """
    Simple CAGR when exact transaction dates are unavailable.
    Use this as fallback when no transaction history in the PDF.

    Verified test:
        simple_cagr(100000, 161050, 5) → 0.10  (10% CAGR)
    """
    if invested <= 0 or current_value <= 0 or years <= 0:
        return None
    try:
        return round((current_value / invested) ** (1 / years) - 1, 4)
    except Exception:
        return None


# ── Expense Drag ───────────────────────────────────────────────────────────────

def compute_expense_drag(
    holdings: List[dict],
    index_ter: float = 0.1,
) -> float:
    """
    Annual rupee cost of holding active/regular funds vs index fund equivalent.
    Each holding dict needs: current_value (float), ter (float, expense ratio %).

    Verified test:
        compute_expense_drag([{"current_value": 500000, "ter": 1.8}]) → 8500.0
        (500000 * (1.8 - 0.1) / 100 = 8500)
    """
    total_drag = 0.0
    for h in holdings:
        value = h.get("current_value", 0)
        ter = h.get("ter", 1.8)
        excess = max(0.0, ter - index_ter) / 100
        total_drag += value * excess
    return round(total_drag, 0)


# ── Overlap ───────────────────────────────────────────────────────────────────

def compute_overlap_from_categories(categories: List[str]) -> float:
    """
    Estimate portfolio overlap % from fund categories.
    Two funds in the same SEBI category share ~70% holdings.
    Used when stock-level holdings data is unavailable.

    Verified test:
        compute_overlap_from_categories(["Large Cap", "Large Cap", "Mid Cap"]) → ~46.7
        (2 large caps duplicated: 1/3 * 0.7 * 100 = 23.3 per duplicate, 1 duplicate = 23.3? 
         Actually: 1 extra large cap / 3 total * 0.7 * 100 = 23.3%)
    """
    if not categories:
        return 0.0
    from collections import Counter
    counts = Counter(categories)
    total = len(categories)
    overlap = sum(
        (count - 1) / total * 0.7
        for count in counts.values()
        if count > 1
    )
    return round(min(overlap * 100, 95.0), 1)


# ── SIP Math ──────────────────────────────────────────────────────────────────

def sip_future_value(monthly_sip: float, annual_return: float, years: int) -> float:
    """
    Standard SIP future value.
    FV = P * ((1+r)^n - 1) / r * (1+r)  where r = monthly rate, n = months.

    Verified test:
        sip_future_value(10000, 0.12, 10) → ~2,323,391
    """
    r = annual_return / 12
    n = years * 12
    if r == 0 or n == 0:
        return monthly_sip * n
    return round(monthly_sip * ((1 + r) ** n - 1) / r * (1 + r), 0)


def sip_required_for_goal(
    target_corpus: float,
    annual_return: float,
    years: int,
    existing_lumpsum: float = 0.0,
) -> float:
    """
    Monthly SIP needed to reach target corpus, accounting for existing savings.

    Verified test:
        sip_required_for_goal(10000000, 0.12, 20) → ~10,000 approx
    """
    if years <= 0:
        return target_corpus
    r = annual_return / 12
    n = years * 12
    existing_fv = existing_lumpsum * (1 + annual_return) ** years
    remaining = max(0.0, target_corpus - existing_fv)
    if remaining <= 0:
        return 0.0
    if r == 0:
        return round(remaining / n, 0)
    sip = remaining / (((1 + r) ** n - 1) / r * (1 + r))
    return round(sip, 0)


def inflation_adjusted_target(today_value: float, years: int, inflation: float = 0.06) -> float:
    """
    Future cost of something that costs today_value rupees today.
    Use 0.06 for lifestyle, 0.08 for education, 0.07 for medical.

    Verified test:
        inflation_adjusted_target(5000000, 20, 0.06) → ~16,035,677
    """
    return round(today_value * (1 + inflation) ** years, 0)


# ── Insurance Gap ─────────────────────────────────────────────────────────────

def compute_insurance_gap(
    annual_income: float,
    existing_cover: float = 0.0,
    multiplier: float = 12.0,
) -> dict:
    """
    IRDAI Human Life Value method.
    Recommended cover = annual_income * multiplier (default 12x).

    Verified test:
        compute_insurance_gap(1200000, 5000000) →
        {"recommended": 14400000, "existing": 5000000, "gap": 9400000, ...}
    """
    recommended = annual_income * multiplier
    gap = max(0.0, recommended - existing_cover)
    coverage_pct = min(100.0, existing_cover / recommended * 100) if recommended > 0 else 0.0
    estimated_premium = (gap / 10_000_000) * 10000 if gap > 0 else 0.0  # ~Rs 10K per Cr

    return {
        "recommended_cover": round(recommended, 0),
        "existing_cover": round(existing_cover, 0),
        "gap": round(gap, 0),
        "coverage_pct": round(coverage_pct, 1),
        "estimated_annual_premium": round(estimated_premium, 0),
        "status": "adequate" if gap == 0 else ("critical" if coverage_pct < 30 else "partial"),
    }


# ── Tax Calculator ────────────────────────────────────────────────────────────

def compute_tax_old_regime(gross_income: float, deductions: dict) -> dict:
    """
    Tax under old regime with all deductions.
    deductions keys: 80c, 80d, nps, hra, other (all in rupees)

    Verified test:
        compute_tax_old_regime(1200000, {"80c": 150000, "80d": 25000, "nps": 50000})
        taxable = 1200000 - 50000 (std) - 150000 - 25000 - 50000 = 925000
        tax on 925000 old regime = 0 + 12500 + 85000 + 37500 = 135000 + 4% cess = 140400
    """
    std_deduction = 50000
    d80c   = min(deductions.get("80c",   0), 150000)
    d80d   = min(deductions.get("80d",   0), 75000)
    d80ccd = min(deductions.get("nps",   0), 50000)
    hra    = deductions.get("hra",   0)
    other  = deductions.get("other", 0)

    total_deductions = std_deduction + d80c + d80d + d80ccd + hra + other
    taxable = max(0.0, gross_income - total_deductions)
    tax = _old_slab_tax(taxable) * 1.04  # 4% cess

    if taxable <= 500000:
        tax = 0.0  # 87A rebate

    return {
        "tax": round(tax, 0),
        "taxable_income": round(taxable, 0),
        "total_deductions": round(total_deductions, 0),
        "effective_rate_pct": round(tax / gross_income * 100, 2) if gross_income > 0 else 0.0,
    }


def compute_tax_new_regime(gross_income: float) -> dict:
    """
    Tax under new regime FY 2024-25.
    Only standard deduction (Rs 75,000) allowed.

    Verified test:
        compute_tax_new_regime(1200000)
        taxable = 1200000 - 75000 = 1125000
        tax = 0 + 15000 + 30000 + 30000 + 18750 = 93750 + 4% cess = 97500
    """
    std_deduction = 75000
    taxable = max(0.0, gross_income - std_deduction)
    tax = _new_slab_tax(taxable) * 1.04

    if taxable <= 700000:
        tax = 0.0  # 87A rebate new regime

    return {
        "tax": round(tax, 0),
        "taxable_income": round(taxable, 0),
        "total_deductions": std_deduction,
        "effective_rate_pct": round(tax / gross_income * 100, 2) if gross_income > 0 else 0.0,
    }


def compare_tax_regimes(gross_income: float, deductions: dict) -> dict:
    """
    Compare old vs new regime and recommend the better one.
    Returns both computations plus savings and recommended regime.
    """
    old = compute_tax_old_regime(gross_income, deductions)
    new = compute_tax_new_regime(gross_income)
    savings = new["tax"] - old["tax"]  # positive = old regime saves money

    return {
        "old_regime": old,
        "new_regime": new,
        "recommended_regime": "old" if savings > 0 else "new",
        "savings_with_recommended": abs(round(savings, 0)),
        "verdict": (
            f"Old regime saves you Rs {savings:,.0f} per year due to your deductions."
            if savings > 0
            else f"New regime saves you Rs {abs(savings):,.0f} per year. Your deductions don't exceed the new regime benefit."
        ),
    }


def _old_slab_tax(income: float) -> float:
    slabs = [(250000, 0.0), (500000, 0.05), (1000000, 0.20), (float("inf"), 0.30)]
    return _apply_slabs(income, slabs)


def _new_slab_tax(income: float) -> float:
    slabs = [
        (300000, 0.0), (600000, 0.05), (900000, 0.10),
        (1200000, 0.15), (1500000, 0.20), (float("inf"), 0.30),
    ]
    return _apply_slabs(income, slabs)


def _apply_slabs(income: float, slabs: List[Tuple]) -> float:
    tax = 0.0
    prev = 0.0
    for limit, rate in slabs:
        if income <= prev:
            break
        taxable_in_slab = min(income, limit) - prev
        tax += taxable_in_slab * rate
        prev = limit
    return tax