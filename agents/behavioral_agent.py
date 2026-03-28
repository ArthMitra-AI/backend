"""
Behavioral Money Mirror Agent.
2-stage Gemini Flash pipeline.
Stage 1: extract transactions from raw SMS text
Stage 2: behavioral analysis — leaks, pattern, subscriptions

Raw SMS text is NEVER written to DB. Processed in memory only.
Only the structured insights are stored.
"""
import json
import logging
import uuid
from collections import defaultdict
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


async def run_behavioral_analysis(
    sms_text: str,
    monthly_income: float,
    age: int,
    goals: str = "Build wealth and retire early",
    language: str = "en",
) -> dict:
    """
    Entry point called by backend dev's FastAPI route.
    sms_text: raw pasted SMS (max 50KB, enforced at route level)
    Returns dict matching BehavioralAnalysisResponse schema.
    """
    analysis_id = str(uuid.uuid4())[:8].upper()
    prompts = _load_prompts()

    # ── Stage 1: Extract transactions ─────────────────────────────────────
    logger.info(f"[{analysis_id}] Stage 1: extracting transactions")

    extract_system = prompts.get("behavioral_extract_v1.0.0", {}).get("system", "")
    extract_prompt = (
        f"Extract all financial transactions from these SMS messages:\n\n"
        f"{sms_text[:40000]}"
    )

    from core.llm import call_llm_json
    transactions = await call_llm_json(extract_prompt, system=extract_system, use_cache=True)

    if not isinstance(transactions, list):
        transactions = []

    logger.info(f"[{analysis_id}] Extracted {len(transactions)} transactions")

    # ── Stage 2: Behavioral analysis ──────────────────────────────────────
    logger.info(f"[{analysis_id}] Stage 2: behavioral analysis")

    period_days = _period_days(transactions)

    analyze_system = prompts.get("behavioral_analyze_v1.0.0", {}).get("system", "")

    if language == "hi":
        analyze_system += "\n\nIMPORTANT: Respond ENTIRELY in Hindi."
    elif language == "ta":
        analyze_system += "\n\nIMPORTANT: Respond ENTIRELY in Tamil."
    elif language == "te":
        analyze_system += "\n\nIMPORTANT: Respond ENTIRELY in Telugu."
    elif language == "bn":
        analyze_system += "\n\nIMPORTANT: Respond ENTIRELY in Bengali."

    analyze_prompt = f"""Analyze spending behavior:

USER PROFILE:
- Monthly income: Rs {monthly_income:,.0f}
- Age: {age}
- Goals: {goals}
- Period covered: {period_days} days

TRANSACTIONS ({len(transactions)} total):
{json.dumps(transactions[:200], default=str)}

Provide insights using ONLY data from these transactions. No invented numbers."""

    try:
        analysis = await call_llm_json(analyze_prompt, system=analyze_system, use_cache=True)
        if not isinstance(analysis, dict):
            analysis = {}
    except Exception as e:
        logger.warning(f"[{analysis_id}] Stage 2 LLM failed: {e} — using computed fallback")
        analysis = _compute_fallback(transactions, monthly_income)

    # ── Fill day-of-week data if missing ──────────────────────────────────
    if not analysis.get("behavioral_pattern", {}).get("day_of_week_data"):
        dow = _dow_data(transactions)
        if "behavioral_pattern" not in analysis:
            analysis["behavioral_pattern"] = {}
        analysis["behavioral_pattern"]["day_of_week_data"] = dow

    savings_potential = _total_savings(analysis)
    compound_3yr      = _compound(savings_potential * 12, 0.12, 3)

    return {
        "analysis_id":                  analysis_id,
        "period_days":                  period_days,
        "total_monthly_spend":          _monthly_spend(transactions, period_days),
        "top_leaks":                    analysis.get("top_leaks", [])[:3],
        "behavioral_pattern":           analysis.get("behavioral_pattern", _default_pattern(transactions)),
        "subscriptions":                analysis.get("subscriptions", []),
        "total_monthly_savings_potential": savings_potential,
        "compound_savings_3yr":         compound_3yr,
        "ai_narrative":                 analysis.get("ai_narrative", "Analysis complete."),
        "language":                     language,
        "prompt_version":               "behavioral_analyze_v1.0.0",
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _period_days(transactions: list) -> int:
    dates = []
    for tx in transactions:
        d = tx.get("date")
        if isinstance(d, str):
            try:
                dates.append(datetime.fromisoformat(d).date())
            except Exception:
                pass
        elif isinstance(d, date):
            dates.append(d)
    if len(dates) < 2:
        return 30
    return max(1, (max(dates) - min(dates)).days)


def _monthly_spend(transactions: list, period_days: int) -> float:
    total = sum(float(tx.get("amount", 0)) for tx in transactions if tx.get("type") == "debit")
    return round(total / max(period_days, 1) * 30, 0)


def _dow_data(transactions: list) -> list:
    totals  = defaultdict(float)
    counts  = defaultdict(int)
    for tx in transactions:
        if tx.get("type") != "debit":
            continue
        dow = tx.get("day_of_week")
        if dow is not None and 0 <= int(dow) <= 6:
            totals[int(dow)]  += float(tx.get("amount", 0))
            counts[int(dow)]  += 1
    return [
        {"day": DAYS[i], "avg_spend": round(totals[i] / max(counts[i], 1), 0)}
        for i in range(7)
    ]


def _total_savings(analysis: dict) -> float:
    s = sum(float(l.get("monthly_amount", 0)) * 0.5 for l in analysis.get("top_leaks", []))
    s += sum(
        float(sub.get("monthly_amount", 0))
        for sub in analysis.get("subscriptions", [])
        if sub.get("cancel_recommendation")
    )
    return round(s, 0)


def _compound(annual: float, rate: float, years: int) -> float:
    if annual <= 0:
        return 0.0
    r, n = rate / 12, years * 12
    return round((annual / 12) * ((1 + r) ** n - 1) / r * (1 + r), 0)


def _default_pattern(transactions: list) -> dict:
    return {
        "pattern_name": "Insufficient Data",
        "description":  "Not enough transactions to identify a clear behavioral pattern.",
        "trigger":      "Unknown",
        "annual_cost":  0,
        "day_of_week_data": _dow_data(transactions),
    }


def _compute_fallback(transactions: list, monthly_income: float) -> dict:
    """Pure Python fallback when LLM fails — computes from raw transaction data."""
    by_cat   = defaultdict(float)
    by_merch = defaultdict(list)
    subs     = {}

    for tx in transactions:
        if tx.get("type") != "debit":
            continue
        cat     = tx.get("category", "other")
        merchant = tx.get("merchant", "Unknown")
        amount   = float(tx.get("amount", 0))
        by_cat[cat]       += amount
        by_merch[cat].append(merchant)
        if cat == "subscription":
            subs[merchant] = subs.get(merchant, 0) + amount

    top_leaks = []
    for cat, total in sorted(by_cat.items(), key=lambda x: -x[1])[:3]:
        monthly = total * 30 / 30
        pct     = round(monthly / monthly_income * 100, 1) if monthly_income > 0 else 0
        top_leaks.append({
            "category":         cat.replace("_", " ").title(),
            "monthly_amount":   round(monthly, 0),
            "pct_of_income":    pct,
            "merchant_examples": list(set(by_merch[cat]))[:3],
            "goal_impact":      f"Redirecting half adds Rs {round(monthly*0.5*12, 0):,.0f}/year to investments",
        })

    subscriptions = [
        {
            "name":                 name,
            "monthly_amount":       round(amt / 3, 0),
            "last_used_days_ago":   None,
            "cancel_recommendation": amt > 200,
            "compound_savings_3yr": _compound(amt * 4, 0.12, 3),
        }
        for name, amt in subs.items()
    ]

    return {
        "top_leaks":          top_leaks,
        "behavioral_pattern": _default_pattern([]),
        "subscriptions":      subscriptions,
        "ai_narrative":       "Analysis based on transaction data.",
    }


def _load_prompts() -> dict:
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", "prompts.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Could not load prompts.json: {e}")
        return {}