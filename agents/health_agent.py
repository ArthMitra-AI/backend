import json
import logging
from typing import List

logger = logging.getLogger(__name__)


async def run_health_score(
    answers: List[dict],
    monthly_income: float,
    monthly_expenses: float,
    age: int,
    dependents: int = 0,
    language: str = "en",
) -> dict:
    """Entry point called by backend dev's FastAPI route."""
    a = {item["question_id"]: item for item in answers}

    dims = []

    # Emergency Fund
    months = float((a.get("emergency_months") or {}).get("value") or 0)
    s = _s_emergency(months)
    dims.append({"name": "Emergency Fund", "score": s, "status": _status(s),
        "key_finding": f"You have {months:.1f} months saved. Benchmark: 6 months minimum.",
        "action": _act_emergency(months, monthly_expenses)})

    # Insurance
    life_cr     = float((a.get("life_cover_cr")     or {}).get("value") or 0)
    health_lakh = float((a.get("health_cover_lakh") or {}).get("value") or 0)
    annual      = monthly_income * 12
    s = _s_insurance(life_cr, health_lakh, annual)
    dims.append({"name": "Insurance", "score": s, "status": _status(s),
        "key_finding": f"Life cover: Rs {life_cr:.1f}Cr (recommended Rs {annual*10/1e7:.1f}Cr+). Health: Rs {health_lakh:.0f}L (min Rs 5L).",
        "action": _act_insurance(life_cr, health_lakh, annual)})

    # Investments
    sav_rate = float((a.get("savings_rate_pct") or {}).get("value") or 0)
    s = _s_investments(sav_rate)
    dims.append({"name": "Investments", "score": s, "status": _status(s),
        "key_finding": f"You save {sav_rate:.0f}% of income. Target: 20%+ for financial independence.",
        "action": _act_investments(sav_rate, monthly_income)})

    # Debt
    emi_pct = float((a.get("emi_pct") or {}).get("value") or 0)
    s = _s_debt(emi_pct)
    dims.append({"name": "Debt", "score": s, "status": _status(s),
        "key_finding": f"EMIs are {emi_pct:.0f}% of income. RBI guideline: keep below 40%.",
        "action": _act_debt(emi_pct, monthly_income)})

    # Tax
    tax_ans = (a.get("tax_deductions_claimed") or {}).get("answer", "No")
    missed  = _missed_deductions(tax_ans, monthly_income, dependents)
    s = _s_tax(tax_ans)
    dims.append({"name": "Tax Efficiency", "score": s, "status": _status(s),
        "key_finding": f"Estimated missed deductions: Rs {missed:,.0f}/year.",
        "action": f"Claim 80C (Rs 1.5L) + NPS 80CCD(1B) (Rs 50K) + 80D health (Rs 25K) — saves Rs {missed*0.3:,.0f} in tax."})

    # Retirement
    ret_ans = (a.get("retirement_savings") or {}).get("answer", "No")
    s = _s_retirement(ret_ans, age, sav_rate)
    dims.append({"name": "Retirement", "score": s, "status": _status(s),
        "key_finding": _ret_finding(ret_ans, age),
        "action": _act_retirement(ret_ans, age, monthly_income)})

    weights     = [0.20, 0.20, 0.20, 0.15, 0.10, 0.15]
    total_score = int(sum(d["score"] * w for d, w in zip(dims, weights)))
    priority    = [d["action"] for d in sorted(dims, key=lambda x: x["score"])[:3]]
    summary, citations = await _ai_summary(total_score, dims, monthly_income, age, language)

    return {
        "total_score":      total_score,
        "grade":            _grade(total_score),
        "dimensions":       dims,
        "priority_actions": priority,
        "summary":          summary,
        "citations":        citations,
        "confidence":       "HIGH",
        "prompt_version":   "health_score_v1.0.0",
    }


async def _ai_summary(total_score, dims, income, age, language):
    try:
        from core.llm import call_llm
        from core.rag import get_rag_context

        prompts = _load_prompts()
        system  = prompts.get("health_score_v1.0.0", {}).get("system", "")
        rag, citations = get_rag_context(
            "financial health emergency fund insurance investment India",
            collections=["sebi_docs", "insurance_docs", "rbi_docs"],
        )

        lang_note = ""
        if language == "hi": lang_note = "Respond ENTIRELY in Hindi."
        elif language != "en": lang_note = f"Respond ENTIRELY in {language}."

        dim_txt = "\n".join(f"- {d['name']}: {d['score']}/100 — {d['key_finding']}" for d in dims)

        prompt = (
            f"A {age}-year-old Indian earning Rs {income:,.0f}/month scored {total_score}/100.\n"
            f"Dimension scores:\n{dim_txt}\n"
            f"Regulatory context:\n{rag}\n\n"
            f"Write a 2-3 sentence summary: overall assessment, single most urgent fix with Rs number, one encouraging note.\n"
            f"{lang_note}\nReturn only the summary text."
        )

        summary = await call_llm(prompt, system=system, use_cache=True)
        return summary.strip(), citations

    except Exception as e:
        logger.warning(f"AI summary failed: {e}")
        worst = min(dims, key=lambda d: d["score"])
        return (
            f"Your Money Health Score is {total_score}/100. "
            f"Most urgent: {worst['name']} — {worst['action']}",
            []
        )


# ── Scoring ───────────────────────────────────────────────────────────────────

def _s_emergency(months):
    if months >= 6: return 100
    if months >= 3: return 65
    if months >= 1: return 30
    return 5

def _s_insurance(life_cr, health_lakh, annual_income):
    rec  = max(0.1, annual_income * 10 / 1e7)
    ls   = min(100, life_cr / rec * 100) if life_cr > 0 else 0
    hs   = min(100, health_lakh / 5 * 100) if health_lakh > 0 else 0
    return int(ls * 0.6 + hs * 0.4)

def _s_investments(rate):
    if rate >= 30: return 100
    if rate >= 20: return 75
    if rate >= 10: return 45
    if rate >  0:  return 20
    return 0

def _s_debt(emi_pct):
    if emi_pct == 0:   return 100
    if emi_pct <= 20:  return 85
    if emi_pct <= 35:  return 60
    if emi_pct <= 50:  return 30
    return 5

def _s_tax(ans):
    a = ans.lower()
    if "maximize" in a or "yes" in a: return 100
    if "some" in a: return 50
    return 10

def _s_retirement(ans, age, sav_rate):
    a    = ans.lower()
    base = 70 if ("regularly" in a or "yes" in a) else (40 if "some" in a else 0)
    return min(100, base + min(30, int(sav_rate / 30 * 30)) if base > 0 else 0)

def _status(s):
    if s >= 85: return "excellent"
    if s >= 65: return "good"
    if s >= 40: return "fair"
    if s >= 20: return "poor"
    return "critical"

def _grade(s):
    if s >= 90: return "A+"
    if s >= 80: return "A"
    if s >= 70: return "B+"
    if s >= 60: return "B"
    if s >= 50: return "C"
    if s >= 35: return "D"
    return "F"


# ── Action text ───────────────────────────────────────────────────────────────

def _act_emergency(months, expenses):
    gap = max(0, 6 - months)
    amt = gap * expenses
    if amt <= 0: return "Emergency fund healthy. Keep it in a liquid MF for ~7% returns vs 3.5% savings account."
    return f"Need Rs {amt:,.0f} more ({gap:.1f} months). Automate Rs {amt/12:,.0f}/month to a liquid mutual fund."

def _act_insurance(life_cr, health_lakh, annual_income):
    rec  = round(annual_income * 10 / 1e7, 1)
    bits = []
    if life_cr < rec:
        gap = rec - life_cr
        bits.append(f"Buy Rs {gap:.1f}Cr term cover (~Rs {gap*10000:,.0f}/year per IRDAI HLV norms)")
    if health_lakh < 5:
        bits.append("Add Rs 5L family floater health policy (~Rs 12,000–18,000/year)")
    return ". ".join(bits) if bits else "Insurance adequate per IRDAI Human Life Value norms."

def _act_investments(rate, income):
    gap = max(0, 20 - rate)
    amt = income * gap / 100
    if amt <= 0: return "Strong savings rate. Ensure at least 60% in equity for inflation-beating returns."
    return f"Increase SIP by Rs {amt:,.0f}/month to reach 20% savings rate. Start with a Nifty 50 index fund."

def _act_debt(emi_pct, income):
    if emi_pct <= 20: return "Debt well-managed. Avoid new personal loans or credit card EMIs."
    if emi_pct <= 40: return "Manageable. Don't add new EMIs. Prepay credit card debt first (highest interest rate)."
    excess = (emi_pct - 40) / 100 * income
    return f"EMI burden critical. Reduce by Rs {excess:,.0f}/month. Order: credit card → personal loan → car → home."

def _missed_deductions(ans, income, dependents):
    annual = income * 12
    if annual < 500000: return 0
    a = ans.lower()
    m = 0 if "maximize" in a else (75000 if "some" in a else 225000)
    if dependents > 0: m += 25000
    return m

def _ret_finding(ans, age):
    a = ans.lower()
    if "regularly" in a or "yes" in a:
        return f"Good — retirement savings in place at {age}. Target EPF + NPS + equity MF combination."
    years = max(0, 60 - age)
    return f"No retirement savings at {age}. {years} years to go. Starting now vs 5 years later = 40% more corpus."

def _act_retirement(ans, age, income):
    a = ans.lower()
    if "regularly" in a or "yes" in a:
        return "Increase NPS contribution to claim full Rs 50,000 deduction under 80CCD(1B) — saves Rs 15,000 tax."
    sip = income * 0.10
    n   = max(1, 60 - age)
    r   = 0.01   # 12% annual / 12 months
    fv  = round(sip * ((1 + r) ** (n * 12) - 1) / r * (1 + r), 0)
    return f"Start retirement SIP of Rs {sip:,.0f}/month (10% of income). Grows to Rs {fv:,.0f} by age 60 at 12% returns."


def _load_prompts():
    import os, json
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", "prompts.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"prompts.json load failed: {e}")
        return {}