"""
AI Chat Mentor Agent.
Conversational financial Q&A in Hindi and English.
Portfolio-aware — uses user's actual financial profile.
No LangGraph needed — streaming conversation with memory.
"""
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


async def run_chat(
    message: str,
    conversation_history: list,
    user_profile: dict,
    language: str = "en",
) -> dict:
    """
    Entry point called by backend dev's FastAPI route.

    Args:
        message: Current user message
        conversation_history: List of {role, content} dicts — last 20 messages
        user_profile: {
            age, monthly_income, monthly_expenses,
            risk_profile, goals,
            portfolio_summary (optional),
            health_score (optional),
        }
        language: en | hi | ta | te | bn

    Returns:
        {response, follow_up_questions, citations, language, prompt_version}
    """
    prompts = _load_prompts()
    system_base = prompts.get("chat_mentor_v1.0.0", {}).get("system", "")

    # Build profile context
    profile_context = _build_profile_context(user_profile)

    # Language instruction
    lang_instruction = _lang_instruction(language, message)

    # RAG context based on user's question
    rag_context, citations = _get_relevant_rag(message)

    # Full system prompt
    system = (
        system_base
        + "\n\n## USER FINANCIAL PROFILE (use these numbers in your answer):\n"
        + profile_context
        + "\n\n## REGULATORY CONTEXT:\n"
        + rag_context
        + ("\n\n" + lang_instruction if lang_instruction else "")
    )

    # Build messages for LLM
    messages_for_llm = _build_messages(conversation_history, message)

    from core.llm import call_llm

    # Build a single prompt combining history
    history_text = ""
    for msg in conversation_history[-10:]:  # last 10 messages for context
        role = msg.get("role", "user")
        content = msg.get("content", "")
        history_text += f"\n{role.upper()}: {content}"

    full_prompt = (
        f"Conversation so far:{history_text}\n\n"
        f"USER: {message}\n\n"
        f"Respond as ArthMitra AI financial advisor."
    )

    response_text = await call_llm(
        full_prompt,
        system=system,
        use_cache=False,  # Never cache chat — every conversation is unique
        temperature=0.4,
    )

    # Generate follow-up questions
    follow_ups = await _generate_follow_ups(message, response_text, user_profile, language)

    return {
        "response":           response_text,
        "follow_up_questions": follow_ups,
        "citations":          citations,
        "language":           language,
        "prompt_version":     "chat_mentor_v1.0.0",
    }


def _build_profile_context(profile: dict) -> str:
    age = profile.get("age", "unknown")
    income = profile.get("monthly_income", 0)
    expenses = profile.get("monthly_expenses", 0)
    risk = profile.get("risk_profile", "moderate")
    goals = profile.get("goals", "Build wealth")
    health_score = profile.get("health_score")
    portfolio = profile.get("portfolio_summary")

    lines = [
        f"- Age: {age}",
        f"- Monthly income: Rs {income:,.0f}" if income else "- Monthly income: not provided",
        f"- Monthly expenses: Rs {expenses:,.0f}" if expenses else "- Monthly expenses: not provided",
        f"- Risk profile: {risk}",
        f"- Financial goals: {goals}",
    ]

    if health_score:
        lines.append(f"- Money Health Score: {health_score}/100")

    if portfolio:
        xirr = portfolio.get("portfolio_xirr_pct")
        drag = portfolio.get("expense_drag_annual")
        if xirr:
            lines.append(f"- Portfolio XIRR: {xirr}%")
        if drag:
            lines.append(f"- Annual expense drag: Rs {drag:,.0f}")

    return "\n".join(lines)


def _lang_instruction(language: str, message: str) -> str:
    # Auto-detect Hindi from message if not explicitly set
    hindi_chars = sum(1 for c in message if '\u0900' <= c <= '\u097F')
    if hindi_chars > 3:
        language = "hi"

    if language == "hi":
        return "IMPORTANT: The user is communicating in Hindi. Respond ENTIRELY in Hindi. Do not use any English words except technical terms like SIP, XIRR, NAV, EMI."
    elif language == "ta":
        return "IMPORTANT: Respond ENTIRELY in Tamil."
    elif language == "te":
        return "IMPORTANT: Respond ENTIRELY in Telugu."
    elif language == "bn":
        return "IMPORTANT: Respond ENTIRELY in Bengali."
    return ""


def _get_relevant_rag(message: str) -> tuple:
    try:
        from core.rag import get_rag_context
        msg_lower = message.lower()

        if any(w in msg_lower for w in ["tax", "80c", "80d", "nps", "regime", "deduction", "itr"]):
            return get_rag_context(message, collections=["tax_docs"])

        if any(w in msg_lower for w in ["insurance", "term", "health", "cover", "irdai"]):
            return get_rag_context(message, collections=["insurance_docs"])

        if any(w in msg_lower for w in ["mutual fund", "sip", "xirr", "expense ratio", "index", "overlap"]):
            return get_rag_context(message, collections=["sebi_docs"])

        if any(w in msg_lower for w in ["inflation", "repo rate", "fd", "rbi", "interest rate"]):
            return get_rag_context(message, collections=["rbi_docs"])

        # General question — search all
        return get_rag_context(message)

    except Exception as e:
        logger.warning(f"RAG failed in chat: {e}")
        return "", []


def _build_messages(history: list, current_message: str) -> list:
    messages = []
    for msg in history[-10:]:
        role = msg.get("role", "user")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": msg.get("content", "")})
    messages.append({"role": "user", "content": current_message})
    return messages


async def _generate_follow_ups(
    user_message: str,
    ai_response: str,
    profile: dict,
    language: str,
) -> list:
    """Generate 3 contextual follow-up questions."""
    try:
        from core.llm import call_llm

        lang_note = "Generate questions in Hindi." if language == "hi" else ""

        prompt = (
            f"User asked: {user_message}\n"
            f"AI answered: {ai_response[:300]}\n\n"
            f"Generate exactly 3 short follow-up questions the user might want to ask next. "
            f"Make them specific to this conversation, not generic. "
            f"{lang_note}"
            f"Return as JSON array of 3 strings. Example: [\"question 1\", \"question 2\", \"question 3\"]"
        )

        from core.llm import call_llm_json
        result = await call_llm_json(prompt, use_cache=False)
        if isinstance(result, list):
            return result[:3]
        return []

    except Exception as e:
        logger.warning(f"Follow-up generation failed: {e}")
        return [
            "How much should I invest monthly for my goals?",
            "What tax deductions am I missing?",
            "How do I start investing in mutual funds?",
        ]


def _load_prompts() -> dict:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", "prompts.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"prompts.json load failed: {e}")
        return {}