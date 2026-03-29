"""
Parses Form 16 or takes manual salary inputs.
Compares old vs new regime with exact numbers.
Identifies missed deductions with IT Act citations.
"""
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


async def run_tax_analysis(
    form16_data: Optional[dict] = None,
    manual_inputs: Optional[dict] = None,
    language: str = "en",
) -> dict:
    """
    Entry point called by backend dev's FastAPI route.

    Either form16_data (from pdf parser) or manual_inputs must be provided.

    manual_inputs schema:
    {
        "gross_salary": float,
        "deductions": {
            "80c": float,
            "80d": float,
            "nps": float,
            "hra": float,
            "other": float
        },
        "tds_deducted": float
    }
    """
    # Use form16 data if available, else manual inputs, else demo data
    if form16_data:
        data = form16_data
    elif manual_inputs:
        data = manual_inputs
    else:
        from tools.form16_parser import generate_demo_form16
        data = generate_demo_form16()
        logger.info("Using demo Form 16 data")

    gross = data.get("gross_salary", 0)
    deductions = data.get("deductions", {})
    tds = data.get("tds_deducted", 0)

    # Step 1 — Compute tax both regimes using our calculation engine
    from tools.finance_calc import compute_tax_old_regime, compute_tax_new_regime, compare_tax_regimes

    comparison = compare_tax_regimes(gross, deductions)
    old = comparison["old_regime"]
    new = comparison["new_regime"]

    # Step 2 — Identify missed deductions
    missed = _identify_missed_deductions(gross, deductions)

    # Step 3 — AI narrative via LLM
    narrative = await _generate_tax_narrative(
        gross, old, new, missed, comparison, tds, language
    )

    total_missed_savings = sum(m["estimated_savings"] for m in missed)

    return {
        "gross_income":             gross,
        "old_regime":               old,
        "new_regime":               new,
        "recommended_regime":       comparison["recommended_regime"],
        "savings_with_recommended": comparison["savings_with_recommended"],
        "missed_deductions":        missed,
        "total_missed_savings":     round(total_missed_savings, 0),
        "tds_deducted":             tds,
        "refund_or_due":            round(tds - (old["tax"] if comparison["recommended_regime"] == "old" else new["tax"]), 0),
        "verdict":                  comparison["verdict"],
        "ai_summary":               narrative,
        "citations":                [
            "Income Tax Act 1961 — Sections 80C, 80CCD, 80D",
            "Finance Act 2023 — New Tax Regime FY 2024-25",
        ],
        "prompt_version":           "tax_wizard_v1.0.0",
    }


def _identify_missed_deductions(gross: float, claimed: dict) -> list:
    """
    Check all major deduction sections and flag what's missing.
    Returns list of missed deductions with estimated tax savings.
    """
    missed = []
    annual = gross

    # Determine tax bracket for savings calculation
    if annual > 1000000:
        tax_rate = 0.30
    elif annual > 500000:
        tax_rate = 0.20
    else:
        tax_rate = 0.05

    # 80C check
    claimed_80c = claimed.get("80c", 0)
    if claimed_80c < 150000:
        gap = 150000 - claimed_80c
        missed.append({
            "section":           "80C",
            "description":       "ELSS mutual fund, PPF, or NSC investment",
            "max_limit":         150000,
            "currently_claimed": claimed_80c,
            "additional_possible": gap,
            "estimated_savings": round(gap * tax_rate * 1.04, 0),
            "action":            f"Invest Rs {gap:,.0f} more in ELSS or PPF before March 31 to claim full 80C benefit.",
        })

    # 80CCD(1B) NPS check
    claimed_nps = claimed.get("nps", 0)
    if claimed_nps < 50000:
        gap = 50000 - claimed_nps
        missed.append({
            "section":           "80CCD(1B)",
            "description":       "NPS contribution — additional Rs 50,000 over 80C limit",
            "max_limit":         50000,
            "currently_claimed": claimed_nps,
            "additional_possible": gap,
            "estimated_savings": round(gap * tax_rate * 1.04, 0),
            "action":            f"Open NPS account and contribute Rs {gap:,.0f} to save Rs {round(gap*tax_rate*1.04):,.0f} extra tax.",
        })

    # 80D health insurance check
    claimed_80d = claimed.get("80d", 0)
    if claimed_80d < 25000:
        gap = 25000 - claimed_80d
        missed.append({
            "section":           "80D",
            "description":       "Health insurance premium for self and family",
            "max_limit":         25000,
            "currently_claimed": claimed_80d,
            "additional_possible": gap,
            "estimated_savings": round(gap * tax_rate * 1.04, 0),
            "action":            f"Buy health insurance with Rs {gap:,.0f} annual premium to claim 80D deduction.",
        })

    # HRA check — only if no HRA claimed but likely paying rent
    claimed_hra = claimed.get("hra", 0)
    if claimed_hra == 0 and annual > 600000:
        missed.append({
            "section":           "HRA",
            "description":       "House Rent Allowance exemption if paying rent",
            "max_limit":         None,
            "currently_claimed": 0,
            "additional_possible": None,
            "estimated_savings": round(annual * 0.03, 0),
            "action":            "If you pay rent, submit rent receipts to employer or claim HRA exemption while filing ITR.",
        })

    return missed


async def _generate_tax_narrative(
    gross: float,
    old: dict,
    new: dict,
    missed: list,
    comparison: dict,
    tds: float,
    language: str,
) -> str:
    try:
        from core.llm import call_llm
        from core.rag import get_rag_context

        prompts = _load_prompts()
        system = prompts.get("tax_wizard_v1.0.0", {}).get("system", "")

        rag_ctx, _ = get_rag_context(
            "income tax deduction 80C 80D NPS old new regime",
            collections=["tax_docs"],
        )

        lang_note = ""
        if language == "hi": lang_note = "Respond ENTIRELY in Hindi."
        elif language != "en": lang_note = f"Respond ENTIRELY in {language}."

        missed_text = "\n".join(
            f"- Section {m['section']}: {m['description']} — can save Rs {m['estimated_savings']:,.0f}"
            for m in missed
        )

        refund = tds - (old["tax"] if comparison["recommended_regime"] == "old" else new["tax"])

        prompt = f"""Analyze this Indian salaried employee's tax situation for FY 2024-25:

INCOME AND TAX:
- Gross salary: Rs {gross:,.0f}
- Old regime tax: Rs {old['tax']:,.0f} (effective rate {old['effective_rate_pct']}%)
- New regime tax: Rs {new['tax']:,.0f} (effective rate {new['effective_rate_pct']}%)
- Recommended: {comparison['recommended_regime']} regime saves Rs {comparison['savings_with_recommended']:,.0f}
- TDS already deducted: Rs {tds:,.0f}
- Refund/Due: Rs {refund:,.0f} ({'refund' if refund > 0 else 'additional tax due'})

MISSED DEDUCTIONS:
{missed_text if missed_text else 'No major missed deductions found'}

REGULATORY CONTEXT:
{rag_ctx}

Write a 3-4 sentence summary: regime recommendation with exact savings, top missed deduction to fix, refund status.
{lang_note}
Return only the summary text."""

        return await call_llm(prompt, system=system, use_cache=True)

    except Exception as e:
        logger.warning(f"Tax narrative failed: {e}")
        rec = comparison['recommended_regime']
        savings = comparison['savings_with_recommended']
        return (
            f"Switch to the {rec} tax regime to save Rs {savings:,.0f} this year. "
            f"You have {len(missed)} missed deduction opportunities worth Rs {sum(m['estimated_savings'] for m in missed):,.0f} in additional savings."
        )


def _load_prompts() -> dict:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", "prompts.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"prompts.json load failed: {e}")
        return {}