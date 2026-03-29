import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

_chroma_client = None
_collections: dict = {}
_is_ready = False


REGULATORY_DOCS = {
    "sebi_docs": [
        {
            "id": "sebi_mf_categorization",
            "text": (
                "SEBI Circular SEBI/HO/IMD/DF3/CIR/P/2017/114 — MF Categorization Oct 2017.\n"
                "Large Cap Fund: minimum 80% in top 100 companies by market cap.\n"
                "Mid Cap Fund: minimum 65% in 101st to 250th companies by market cap.\n"
                "Small Cap Fund: minimum 65% in 251st company onwards.\n"
                "Flexi Cap Fund: minimum 65% in equity, no market cap restriction.\n"
                "Index Fund: minimum 95% in securities of the replicated index.\n"
                "ELSS: minimum 80% in equity, 3-year lock-in, qualifies for 80C deduction.\n"
                "Overlap between two funds in the same SEBI category is typically 60-80%.\n"
                "Holding multiple funds from the same category gives false diversification."
            ),
            "source": "SEBI Circular SEBI/HO/IMD/DF3/CIR/P/2017/114",
        },
        {
            "id": "sebi_direct_vs_regular",
            "text": (
                "AMFI Direct Plan Guidelines — introduced January 2013 per SEBI directive.\n"
                "Direct plans have no distributor commission — lower Total Expense Ratio (TER).\n"
                "TER difference between regular and direct: typically 0.5% to 1.5% per year.\n"
                "On a Rs 10 lakh portfolio over 10 years: direct plan saves Rs 2.5 to Rs 5 lakh.\n"
                "All AMCs must disclose TER on their website daily per SEBI regulations.\n"
                "Switching from regular to direct is treated as redemption — check exit load and tax."
            ),
            "source": "SEBI/AMFI Direct Plan Circular 2013",
        },
        {
            "id": "sebi_ia_regulations",
            "text": (
                "SEBI Investment Adviser Regulations 2013.\n"
                "Any person providing investment advice for consideration must register with SEBI.\n"
                "Registered advisers must act as fiduciaries — client's best interest always first.\n"
                "Mandatory disclaimer: Past performance is not indicative of future returns.\n"
                "Advisers must complete risk profiling before recommending any investment.\n"
                "Investors can verify SEBI registration at sebi.gov.in."
            ),
            "source": "SEBI Investment Adviser Regulations 2013",
        },
    ],
    "tax_docs": [
        {
            "id": "it_80c_deductions",
            "text": (
                "Income Tax Act Section 80C — max deduction Rs 1,50,000 per financial year.\n"
                "Eligible: ELSS mutual funds, PPF, NSC, 5-year bank FD, LIC premium, "
                "home loan principal, tuition fees for 2 children, ULIP.\n"
                "PPF: 15-year lock-in, government-declared interest rate (~7.1% current).\n"
                "ELSS: 3-year lock-in, equity exposure, historically highest returns among 80C options.\n"
                "Section 80CCD(1B): Additional Rs 50,000 deduction for NPS contribution — "
                "this is OVER AND ABOVE the Rs 1.5 lakh 80C limit.\n"
                "Section 80D: Rs 25,000 for self and family health insurance. "
                "Rs 50,000 if parents are senior citizens.\n"
                "Standard Deduction: Rs 75,000 under new regime, Rs 50,000 under old regime."
            ),
            "source": "Income Tax Act 1961 — Sections 80C, 80CCD, 80D",
        },
        {
            "id": "it_old_vs_new_regime",
            "text": (
                "Old vs New Tax Regime FY 2024-25.\n"
                "NEW REGIME (default from FY 2023-24):\n"
                "0 to 3L: 0% | 3L to 6L: 5% | 6L to 9L: 10% | 9L to 12L: 15% | "
                "12L to 15L: 20% | above 15L: 30%.\n"
                "New regime: no deductions (80C, 80D, HRA not allowed). "
                "Standard deduction Rs 75,000 allowed. "
                "87A rebate: zero tax if taxable income <= Rs 7 lakh.\n"
                "OLD REGIME:\n"
                "0 to 2.5L: 0% | 2.5L to 5L: 5% | 5L to 10L: 20% | above 10L: 30%.\n"
                "Old regime: all deductions available.\n"
                "87A rebate: zero tax if taxable income <= Rs 5 lakh.\n"
                "Rule of thumb: old regime better if total deductions exceed Rs 3.75 lakh. "
                "Both regimes add 4% Health and Education Cess on tax computed."
            ),
            "source": "Finance Act 2023 — Income Tax Regimes FY 2024-25",
        },
    ],
    "rbi_docs": [
        {
            "id": "rbi_key_rates",
            "text": (
                "RBI Monetary Policy Key Rates 2024.\n"
                "Repo Rate: 6.5% — rate at which RBI lends to commercial banks.\n"
                "Rising repo rate: FD rates rise, bond prices fall, debt fund NAVs fall.\n"
                "Falling repo rate: bond prices rise, debt fund NAVs rise, equities typically rally.\n"
                "CPI inflation target: 4% plus or minus 2%. Current CPI approximately 5%.\n"
                "Real return = nominal return minus inflation.\n"
                "A 7% FD at 5% inflation gives only 2% real return.\n"
                "Equity historically delivers 5 to 8% real return over 10+ year periods.\n"
                "For long-term planning: assume 6% inflation for lifestyle, 7% for education costs."
            ),
            "source": "RBI Monetary Policy Report 2024",
        },
    ],
    "insurance_docs": [
        {
            "id": "irdai_coverage_norms",
            "text": (
                "IRDAI Life Insurance Coverage Norms — Human Life Value Method.\n"
                "Recommended cover = Annual Income multiplied by 10 to 15.\n"
                "For Rs 12 LPA income: minimum Rs 1.2 Crore, recommended Rs 1.5 to 1.8 Crore.\n"
                "Term insurance: pure protection plan, no maturity benefit, lowest premium.\n"
                "Approximate premium: Rs 8,000 to Rs 12,000 per year for Rs 1 Crore cover, "
                "30-year-old non-smoker, 30-year term.\n"
                "Critical illness rider: pays lump sum on diagnosis of 36 listed illnesses.\n"
                "Health insurance minimum: Rs 5 lakh base cover per family.\n"
                "Super top-up plan: cost-effective way to increase health cover.\n"
                "Insurance gap = Recommended HLV cover minus existing total cover."
            ),
            "source": "IRDAI Guidelines on Life and Health Insurance",
        },
    ],
}


def initialize_chromadb() -> int:
    """
    Called by backend dev in FastAPI lifespan startup event.
    Loads all regulatory docs into ChromaDB in-process.
    Returns total chunk count — shown in /api/health endpoint.
    Takes ~20 seconds on cold start (model download + embedding).
    """
    global _chroma_client, _collections, _is_ready

    logger.info("Initializing ChromaDB in-process...")

    try:
        import chromadb
        from chromadb.utils import embedding_functions

        _chroma_client = chromadb.Client()  # ephemeral in-memory — no disk writes

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"  # 22MB CPU-only model
        )

        total_chunks = 0

        for col_name, docs in REGULATORY_DOCS.items():
            col = _chroma_client.get_or_create_collection(
                name=col_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )

            existing_ids = set(col.get()["ids"])
            new_docs = [d for d in docs if d["id"] not in existing_ids]

            if new_docs:
                col.add(
                    ids=[d["id"] for d in new_docs],
                    documents=[d["text"] for d in new_docs],
                    metadatas=[{"source": d["source"]} for d in new_docs],
                )

            _collections[col_name] = col
            total_chunks += len(docs)
            logger.info(f"  {col_name}: {len(docs)} chunks loaded")

        _is_ready = True
        logger.info(f"ChromaDB ready — {total_chunks} total chunks")
        return total_chunks

    except Exception as e:
        logger.error(f"ChromaDB init failed: {e}")
        _is_ready = False
        return 0


def get_rag_context(
    query: str,
    collections: List[str] = None,
    n_results: int = 3,
    threshold: float = 0.72,
) -> Tuple[str, List[str]]:
    """
    Your main RAG function. Call this from agents before building prompts.
    Returns (formatted_context_string, list_of_citation_strings).
    Returns ("", []) if ChromaDB not ready — agents degrade gracefully.
    """
    if not _is_ready:
        logger.warning("ChromaDB not ready — returning empty RAG context")
        return "", []

    if collections is None:
        collections = list(REGULATORY_DOCS.keys())

    results = []

    for col_name in collections:
        if col_name not in _collections:
            continue
        try:
            col = _collections[col_name]
            count = col.count()
            if count == 0:
                continue
            r = col.query(
                query_texts=[query],
                n_results=min(n_results, count),
                include=["documents", "metadatas", "distances"],
            )
            for doc, meta, dist in zip(
                r["documents"][0], r["metadatas"][0], r["distances"][0]
            ):
                similarity = 1 - (dist / 2)  # cosine distance → similarity
                if similarity >= threshold:
                    results.append({
                        "text": doc,
                        "source": meta.get("source", ""),
                        "similarity": round(similarity, 3),
                    })
        except Exception as e:
            logger.warning(f"RAG query error in {col_name}: {e}")

    results.sort(key=lambda x: -x["similarity"])
    top = results[:n_results]

    if not top:
        return "", []

    context = "\n\n---\n\n".join(
        f"[Regulatory Source: {c['source']}]\n{c['text']}" for c in top
    )
    citations = [c["source"] for c in top]

    return context, citations


def get_chromadb_status() -> dict:
    """Called by backend dev in /api/health endpoint."""
    return {
        "ready": _is_ready,
        "collections": len(_collections),
        "total_chunks": sum(
            _collections[c].count() for c in _collections
        ) if _is_ready else 0,
    }