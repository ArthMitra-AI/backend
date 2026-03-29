"""
Financial Independence Retire Early roadmap generator.
Takes age, income, expenses, goals and outputs month-by-month SIP plan.
"""
import json
import logging
import os

logger = logging.getLogger(__name__)


async def run_fire_planner(
    age: int,
    monthly_income: float,
    monthly_expenses: float,
    target_fire_age: int = 45,
    existing_savings: float = 0,
    existing_investments: float = 0,
    existing_life_cover: float = 0,
    language: str = "en",
) -> dict:
    """
    Entry point called by backend dev's FastAPI route.

    Args:
        age: Current age
        monthly_income: Take-home monthly income
        monthly_expenses: Current monthly expenses
        target_fire_age: Age at which user wants to retire (default 45)
        existing_savings: Current liquid savings
        existing_investments: Current investment portfolio value
        existing_life_cover: Current life insurance cover in rupees
        language: Response language
    """
    from tools.finance_calc import (
        sip_future_value,
        sip_required_for_goal,
        inflation_adjusted_target,
        compute_insurance_gap,
    )

    annual_income = monthly_income * 12
    annual_expenses = monthly_expenses * 12
    years_to_fire = max(1, target_fire_age - age)

    # Step 1 — Calculate FIRE corpus needed
    # Target corpus = annual expenses at retirement / 4% safe withdrawal rate
    # Adjust expenses for inflation
    retirement_annual_expenses = inflation_adjusted_target(annual_expenses, years_to_fire, 0.06)
    fire_corpus = retirement_annual_expenses / 0.04  # 4% rule
    inflation_adjusted_corpus = fire_corpus  # Already inflation adjusted

    # Step 2 — Monthly SIP required
    total_existing = existing_savings + existing_investments
    monthly_sip_required = sip_required_for_goal(
        target_corpus=fire_corpus,
        annual_return=0.12,
        years=years_to_fire,
        existing_lumpsum=total_existing,
    )

    current_monthly_savings = monthly_income - monthly_expenses
    sip_gap = max(0, monthly_sip_required - current_monthly_savings)

    # Step 3 — Asset allocation based on years to FIRE
    if years_to_fire >= 15:
        allocation = {"equity_pct": 80, "debt_pct": 15, "gold_pct": 5}
        allocation_label = "Aggressive"
    elif years_to_fire >= 10:
        allocation = {"equity_pct": 60, "debt_pct": 30, "gold_pct": 10}
        allocation_label = "Moderate"
    else:
        allocation = {"equity_pct": 40, "debt_pct": 50, "gold_pct": 10}
        allocation_label = "Conservative"

    # Step 4 — Milestones
    milestones = _generate_milestones(
        age, target_fire_age, monthly_sip_required, fire_corpus, total_existing
    )

    # Step 5 — Insurance gap
    insurance = compute_insurance_gap(annual_income, existing_life_cover)

    # Step 6 — AI summary
    summary = await _generate_fire_summary(
        age, target_fire_age, years_to_fire,
        fire_corpus, monthly_sip_required, sip_gap,
        allocation, allocation_label, insurance, language,
    )

    return {
        "fire_target_corpus":       round(fire_corpus, 0),
        "inflation_adjusted_corpus": round(inflation_adjusted_corpus, 0),
        "target_fire_age":          target_fire_age,
        "years_to_fire":            years_to_fire,
        "monthly_sip_required":     monthly_sip_required,
        "current_savings_monthly":  round(current_monthly_savings, 0),
        "sip_gap":                  sip_gap,
        "asset_allocation":         allocation,
        "allocation_label":         allocation_label,
        "milestones":               milestones,
        "insurance_gap":            insurance,
        "summary":                  summary,
        "assumptions": {
            "equity_return_pct":    12,
            "inflation_pct":        6,
            "safe_withdrawal_rate": 4,
        },
        "citations": [
            "RBI Monetary Policy Report 2024 — inflation assumptions",
            "SEBI Investment Adviser Regulations 2013",
        ],
        "prompt_version": "fire_planner_v1.0.0",
    }


def _generate_milestones(
    current_age: int,
    fire_age: int,
    monthly_sip: float,
    total_corpus: float,
    existing: float,
) -> list:
    """Generate 5 milestone markers on the FIRE journey."""
    from tools.finance_calc import sip_future_value

    milestones = []
    years_to_fire = fire_age - current_age
    checkpoints = [0.25, 0.50, 0.75, 0.90, 1.0]
    labels = [
        "Foundation built — first major corpus milestone",
        "Halfway there — compounding starts accelerating",
        "75% complete — FIRE is now clearly in sight",
        "90% complete — final stretch, stay the course",
        "FIRE achieved — financial independence unlocked",
    ]

    for i, (pct, label) in enumerate(zip(checkpoints, labels)):
        years = int(years_to_fire * pct)
        if years == 0:
            years = 1
        corpus_at_milestone = sip_future_value(monthly_sip, 0.12, years)
        if existing > 0:
            existing_fv = existing * (1.12 ** years)
            corpus_at_milestone += existing_fv

        milestones.append({
            "year":           current_age + years,
            "age":            current_age + years,
            "years_from_now": years,
            "corpus_target":  round(corpus_at_milestone, 0),
            "pct_of_goal":    int(pct * 100),
            "description":    label,
        })

    return milestones


async def _generate_fire_summary(
    age, target_fire_age, years_to_fire,
    corpus, monthly_sip, sip_gap,
    allocation, allocation_label, insurance, language,
) -> str:
    try:
        from core.llm import call_llm
        from core.rag import get_rag_context

        prompts = _load_prompts()
        system = prompts.get("fire_planner_v1.0.0", {}).get("system", "")

        rag_ctx, _ = get_rag_context(
            "FIRE financial independence retire early SIP investment India",
            collections=["rbi_docs", "sebi_docs"],
        )

        lang_note = ""
        if language == "hi": lang_note = "Respond ENTIRELY in Hindi."
        elif language != "en": lang_note = f"Respond ENTIRELY in {language}."

        prompt = f"""Create a personalized FIRE plan summary:

USER PROFILE:
- Current age: {age}
- Target FIRE age: {target_fire_age} ({years_to_fire} years away)
- FIRE corpus needed: Rs {corpus:,.0f}
- Monthly SIP required: Rs {monthly_sip:,.0f}
- Current savings gap: Rs {sip_gap:,.0f}/month
- Asset allocation: {allocation_label} ({allocation['equity_pct']}% equity / {allocation['debt_pct']}% debt / {allocation['gold_pct']}% gold)
- Insurance status: {insurance['status']} (gap: Rs {insurance['gap']:,.0f})

CONTEXT:
{rag_ctx}

Write 3-4 sentences: FIRE date, monthly SIP needed, biggest action to take now, one encouraging note.
{lang_note}
Return only the summary text."""

        return await call_llm(prompt, system=system, use_cache=True)

    except Exception as e:
        logger.warning(f"FIRE summary failed: {e}")
        return (
            f"To retire at {target_fire_age}, you need Rs {corpus:,.0f} corpus in {years_to_fire} years. "
            f"Start a monthly SIP of Rs {monthly_sip:,.0f} today with {allocation_label.lower()} allocation "
            f"({allocation['equity_pct']}% equity). "
            f"Historically equity mutual funds have delivered 12% CAGR over long periods."
        )


def _load_prompts() -> dict:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", "prompts.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"prompts.json load failed: {e}")
        return {}