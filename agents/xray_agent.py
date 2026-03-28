"""
Portfolio X-Ray Agent — LangGraph pipeline.
5 nodes: parse → enrich → calculate → analyze → output
Each node sends WebSocket progress events via the callback.
"""
import json
import logging
import uuid
from datetime import date
from typing import Any, Callable, Optional, TypedDict

logger = logging.getLogger(__name__)


class XRayState(TypedDict):
    scan_id: str
    user_id: str
    pdf_bytes: Optional[bytes]
    parsed_portfolio: Optional[Any]
    enriched_holdings: Optional[list]
    calculations: Optional[dict]
    ai_analysis: Optional[dict]
    final_response: Optional[dict]
    error: Optional[str]
    ws_callback: Optional[Callable]   # backend dev injects this


async def _emit(state: XRayState, event_type: str, message: str, pct: int, data: dict = None):
    cb = state.get("ws_callback")
    if cb:
        try:
            await cb({"type": event_type, "message": message, "progress_pct": pct, "data": data})
        except Exception as e:
            logger.warning(f"WS emit failed: {e}")


# ── Node 1: Parse ─────────────────────────────────────────────────────────────

async def parse_node(state: XRayState) -> XRayState:
    await _emit(state, "PROGRESS", "Reading your MF statement...", 10)
    try:
        from tools.pdf_parser import parse_cams_pdf, generate_demo_portfolio
        if state.get("pdf_bytes"):
            portfolio = parse_cams_pdf(state["pdf_bytes"])
            logger.info(f"Parsed {len(portfolio.funds)} funds")
        else:
            portfolio = generate_demo_portfolio()
            logger.info("Using demo portfolio")
        return {**state, "parsed_portfolio": portfolio}
    except Exception as e:
        return {**state, "error": str(e)}


# ── Node 2: Enrich ────────────────────────────────────────────────────────────

async def enrich_node(state: XRayState) -> XRayState:
    if state.get("error"):
        return state
    await _emit(state, "PROGRESS", "Fetching live NAV from AMFI...", 30)

    portfolio = state["parsed_portfolio"]
    enriched = []

    for fund in portfolio.funds:
        h = {
            "scheme_name":    fund.scheme_name,
            "isin":           fund.isin,
            "units":          fund.units,
            "avg_nav":        fund.avg_nav,
            "invested_amount": fund.invested_amount,
            "current_value":  fund.current_value,
            "category":       fund.category,
            "transactions":   fund.transactions,
            "is_direct":      "direct" in fund.scheme_name.lower(),
        }

        live_nav = await _fetch_nav(fund.isin, fund.scheme_name)
        h["live_nav"] = live_nav or fund.avg_nav
        h["current_value_live"] = round(fund.units * h["live_nav"], 2) if fund.units > 0 else (fund.current_value or 0)
        h["estimated_ter"] = 0.15 if h["is_direct"] else 1.8

        enriched.append(h)

    nifty = await _fetch_nifty_return()
    return {**state, "enriched_holdings": enriched, "benchmark_return": nifty}


# ── Node 3: Calculate ─────────────────────────────────────────────────────────

async def calculate_node(state: XRayState) -> XRayState:
    if state.get("error"):
        return state
    await _emit(state, "PROGRESS", "Computing your true returns (XIRR)...", 55)

    from tools.finance_calc import xirr, simple_cagr, compute_overlap_from_categories, compute_expense_drag

    holdings = state["enriched_holdings"]
    total_invested = sum(h["invested_amount"] for h in holdings)
    total_current  = sum(h["current_value_live"] for h in holdings)

    # Per-fund XIRR
    for h in holdings:
        txns = h.get("transactions", [])
        cfs, ds = [], []
        for tx in sorted(txns, key=lambda x: x.get("date") or date.min):
            d = tx.get("date")
            if not d:
                continue
            amt = tx.get("amount", 0)
            t   = tx.get("type", "").lower()
            if t in ("purchase", "sip", "nfo", "switch in"):
                cfs.append(-amt); ds.append(d)
            elif t in ("redemption", "switch out"):
                cfs.append(amt);  ds.append(d)

        if cfs and h["current_value_live"] > 0:
            cfs.append(h["current_value_live"])
            ds.append(date.today())
            h["fund_xirr"] = xirr(cfs, ds)
        else:
            h["fund_xirr"] = simple_cagr(h["invested_amount"], h["current_value_live"], 2.5)

        h["weight_pct"] = round(h["current_value_live"] / total_current * 100, 1) if total_current > 0 else 0

    portfolio_xirr = simple_cagr(total_invested, total_current, 2.5) if total_invested > 0 else None
    overlap        = compute_overlap_from_categories([h["category"] for h in holdings])
    expense_drag   = compute_expense_drag([{"current_value": h["current_value_live"], "ter": h["estimated_ter"]} for h in holdings])

    calcs = {
        "total_invested":       round(total_invested, 0),
        "total_current_value":  round(total_current,  0),
        "portfolio_xirr":       portfolio_xirr,
        "benchmark_return":     state.get("benchmark_return", 0.141),
        "overlap_score":        overlap,
        "expense_drag_annual":  expense_drag,
    }
    return {**state, "calculations": calcs, "enriched_holdings": holdings}


# ── Node 4: Analyze ───────────────────────────────────────────────────────────

async def analyze_node(state: XRayState) -> XRayState:
    if state.get("error"):
        return state
    await _emit(state, "PROGRESS", "AI generating your rebalancing plan...", 75)

    from core.llm import call_llm_json
    from core.rag import get_rag_context

    calc     = state["calculations"]
    holdings = state["enriched_holdings"]

    rag_ctx, citations = get_rag_context(
        "mutual fund expense ratio overlap SEBI categorization direct plan",
        collections=["sebi_docs"],
    )

    fund_lines = "\n".join(
        f"- {h['scheme_name']}: category={h['category']}, weight={h['weight_pct']}%, "
        f"XIRR={round((h.get('fund_xirr') or 0)*100, 1)}%, direct={h['is_direct']}, TER={h['estimated_ter']}%"
        for h in holdings
    )

    prompt = f"""Analyze this Indian investor's mutual fund portfolio and generate a rebalancing plan.

COMPUTED METRICS:
- Total invested: Rs {calc['total_invested']:,.0f}
- Current value: Rs {calc['total_current_value']:,.0f}
- Portfolio XIRR: {round((calc['portfolio_xirr'] or 0)*100, 1)}%
- Nifty 50 benchmark: {round(calc['benchmark_return']*100, 1)}%
- Overlap score: {calc['overlap_score']}% (critical if above 60%)
- Annual expense drag: Rs {calc['expense_drag_annual']:,.0f}

FUND HOLDINGS:
{fund_lines}

REGULATORY CONTEXT:
{rag_ctx}

Generate a specific rebalancing plan. Use the exact numbers above."""

    prompts_path = _load_prompts()
    system = prompts_path.get("portfolio_xray_v1.0.0", {}).get("system", "")

    try:
        result = await call_llm_json(prompt, system=system, use_pro=True)
    except Exception as e:
        logger.warning(f"analyze_node LLM failed: {e} — using computed fallback")
        result = {
            "ai_summary": (
                f"Your portfolio XIRR is {round((calc['portfolio_xirr'] or 0)*100, 1)}% vs "
                f"Nifty 50's {round(calc['benchmark_return']*100, 1)}%. "
                f"You are losing Rs {calc['expense_drag_annual']:,.0f}/year in expense drag "
                f"and your overlap score of {calc['overlap_score']}% shows dangerous concentration."
            ),
            "rebalancing_plan": [],
            "fund_analysis": [],
            "citations": citations,
        }

    result["citations"]       = list(set(citations + result.get("citations", [])))
    result["prompt_version"]  = "portfolio_xray_v1.0.0"
    return {**state, "ai_analysis": result}


# ── Node 5: Output ────────────────────────────────────────────────────────────

async def output_node(state: XRayState) -> XRayState:
    if state.get("error"):
        await _emit(state, "ERROR", state["error"], 0)
        return state

    await _emit(state, "PROGRESS", "Finalising your X-Ray report...", 95)

    calc     = state["calculations"]
    holdings = state["enriched_holdings"]
    ai       = state["ai_analysis"]

    fund_analysis = [
        {
            "scheme_name":    h["scheme_name"],
            "xirr_pct":       round((h.get("fund_xirr") or 0) * 100, 2),
            "category":       h["category"],
            "ter":            h["estimated_ter"],
            "weight_pct":     h["weight_pct"],
            "is_direct":      h["is_direct"],
            "invested_amount": h["invested_amount"],
            "current_value":  h["current_value_live"],
            "recommendation": _recommendation(h, calc),
        }
        for h in holdings
    ]

    final = {
        "scan_id":              state["scan_id"],
        "total_invested":       calc["total_invested"],
        "current_value":        calc["total_current_value"],
        "portfolio_xirr_pct":   round((calc["portfolio_xirr"] or 0) * 100, 2),
        "benchmark_xirr_pct":   round(calc["benchmark_return"] * 100, 2),
        "overlap_score":        calc["overlap_score"],
        "expense_drag_annual":  calc["expense_drag_annual"],
        "fund_analysis":        fund_analysis,
        "rebalancing_plan":     ai.get("rebalancing_plan", []),
        "ai_summary":           ai.get("ai_summary", ""),
        "citations":            ai.get("citations", []),
        "confidence":           "HIGH" if calc["portfolio_xirr"] else "MEDIUM",
        "prompt_version":       ai.get("prompt_version", "portfolio_xray_v1.0.0"),
    }

    await _emit(state, "COMPLETE", "Your X-Ray is ready!", 100, final)
    return {**state, "final_response": final}


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_xray_graph():
    try:
        from langgraph.graph import StateGraph, END
        g = StateGraph(XRayState)
        g.add_node("parse",     parse_node)
        g.add_node("enrich",    enrich_node)
        g.add_node("calculate", calculate_node)
        g.add_node("analyze",   analyze_node)
        g.add_node("output",    output_node)
        g.set_entry_point("parse")
        g.add_edge("parse",     "enrich")
        g.add_edge("enrich",    "calculate")
        g.add_edge("calculate", "analyze")
        g.add_edge("analyze",   "output")
        g.add_edge("output",    END)
        return g.compile()
    except ImportError:
        logger.error("langgraph not installed")
        return None


async def run_xray_pipeline(
    user_id: str,
    pdf_bytes: Optional[bytes] = None,
    ws_callback: Optional[Callable] = None,
) -> dict:
    """Entry point called by backend dev's FastAPI route."""
    scan_id = str(uuid.uuid4())[:8].upper()
    state: XRayState = {
        "scan_id": scan_id, "user_id": user_id,
        "pdf_bytes": pdf_bytes, "parsed_portfolio": None,
        "enriched_holdings": None, "calculations": None,
        "ai_analysis": None, "final_response": None,
        "error": None, "ws_callback": ws_callback,
    }
    graph = build_xray_graph()
    if graph:
        final = await graph.ainvoke(state)
    else:
        for fn in [parse_node, enrich_node, calculate_node, analyze_node, output_node]:
            state = await fn(state)
            if state.get("error"):
                break
        final = state
    if final.get("error"):
        raise RuntimeError(final["error"])
    return final.get("final_response", {})


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _fetch_nav(isin: Optional[str], name: str) -> Optional[float]:
    try:
        import httpx
        from core.cache import get_cache
        cache = get_cache()
        if isin:
            cached = await cache.get(f"nav:{isin}")
            if cached:
                return float(cached)
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"https://api.mfapi.in/mf/search?q={isin}")
                if r.status_code == 200 and r.json():
                    code = r.json()[0].get("schemeCode")
                    if code:
                        r2 = await client.get(f"https://api.mfapi.in/mf/{code}")
                        if r2.status_code == 200:
                            nav = float(r2.json()["data"][0]["nav"])
                            await cache.set(f"nav:{isin}", str(nav), ex=21600)
                            return nav
    except Exception as e:
        logger.warning(f"NAV fetch failed for {name}: {e}")
    return None


async def _fetch_nifty_return() -> float:
    try:
        from core.cache import get_cache
        cached = await get_cache().get("nifty:cagr")
        if cached:
            return float(cached)
    except Exception:
        pass
    return 0.141  # Nifty 50 ~14.1% long-term CAGR


def _recommendation(h: dict, calc: dict) -> str:
    if not h["is_direct"] and h["estimated_ter"] > 1.0:
        return "switch_to_direct"
    xirr = h.get("fund_xirr") or 0
    if xirr < calc["benchmark_return"] * 0.6:
        return "exit"
    return "hold"


def _load_prompts() -> dict:
    import os, json
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", "prompts.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}