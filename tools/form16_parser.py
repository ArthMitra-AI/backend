"""
Extracts salary and deduction data from Form 16 PDFs.
Processes inmemory only — never written to disk.
"""
import re
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def parse_form16(pdf_bytes: bytes) -> dict:
    """
    Parse Form 16 PDF and extract salary components.
    Returns dict with all fields needed by tax_agent.
    Falls back to zeros for any field not found.
    """
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("Not a valid PDF file")

    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            full_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        return _extract_from_text(full_text)
    except Exception as e:
        logger.error(f"Form 16 parse failed: {e}")
        raise RuntimeError(f"Could not parse Form 16: {e}")


def _extract_from_text(text: str) -> dict:
    def extract_amount(patterns, txt):
        for pattern in patterns:
            m = re.search(pattern, txt, re.IGNORECASE)
            if m:
                return _parse_float(m.group(1))
        return 0.0

    gross_salary = extract_amount([
        r"Gross\s+Salary.*?([\d,]+)",
        r"Total\s+Salary.*?([\d,]+)",
        r"Gross\s+Total\s+Income.*?([\d,]+)",
    ], text)

    basic = extract_amount([
        r"Basic\s+(?:Salary|Pay).*?([\d,]+)",
    ], text)

    hra_received = extract_amount([
        r"House\s+Rent\s+Allowance.*?([\d,]+)",
        r"HRA.*?received.*?([\d,]+)",
    ], text)

    hra_exempt = extract_amount([
        r"HRA.*?exempt.*?([\d,]+)",
        r"Exemption.*?HRA.*?([\d,]+)",
    ], text)

    deduction_80c = extract_amount([
        r"80C.*?([\d,]+)",
        r"Chapter\s+VI.*?80C.*?([\d,]+)",
    ], text)

    deduction_80d = extract_amount([
        r"80D.*?([\d,]+)",
        r"Medical\s+Insurance.*?([\d,]+)",
    ], text)

    deduction_nps = extract_amount([
        r"80CCD.*?([\d,]+)",
        r"NPS.*?([\d,]+)",
        r"National\s+Pension.*?([\d,]+)",
    ], text)

    tds_deducted = extract_amount([
        r"Tax\s+Deducted.*?([\d,]+)",
        r"TDS.*?([\d,]+)",
        r"Total\s+Tax\s+Deducted.*?([\d,]+)",
    ], text)

    employer_name = None
    em = re.search(r"(?:Name\s+of\s+Employer|Employer.*?Name)\s*:?\s*([A-Za-z][A-Za-z\s&.,-]{3,50})", text, re.I)
    if em:
        employer_name = em.group(1).strip()

    employee_name = None
    nm = re.search(r"(?:Name\s+of\s+Employee|Employee.*?Name)\s*:?\s*([A-Za-z][A-Za-z\s]{3,40})", text, re.I)
    if nm:
        employee_name = nm.group(1).strip()

    financial_year = None
    fy = re.search(r"(?:Financial\s+Year|Assessment\s+Year|F\.Y\.?)\s*:?\s*(\d{4}-\d{2,4})", text, re.I)
    if fy:
        financial_year = fy.group(1)

    return {
        "employee_name":    employee_name,
        "employer_name":    employer_name,
        "financial_year":   financial_year or "2024-25",
        "gross_salary":     gross_salary,
        "basic_salary":     basic,
        "hra_received":     hra_received,
        "hra_exempt":       hra_exempt,
        "deductions": {
            "80c":   min(deduction_80c, 150000),
            "80d":   min(deduction_80d, 75000),
            "nps":   min(deduction_nps, 50000),
            "hra":   hra_exempt,
            "other": 0.0,
        },
        "tds_deducted": tds_deducted,
    }


def generate_demo_form16() -> dict:
    """
    Demo Form 16 data for Priya from the demo script.
    Shows missed 80D and NPS deductions — the demo wow moment.
    """
    return {
        "employee_name":  "Priya Sharma",
        "employer_name":  "Tech Solutions Pvt Ltd",
        "financial_year": "2024-25",
        "gross_salary":   1200000,
        "basic_salary":   600000,
        "hra_received":   180000,
        "hra_exempt":     120000,
        "deductions": {
            "80c":   150000,  # Has ELSS — maxed out
            "80d":   0,       # MISSED — no health insurance claimed
            "nps":   0,       # MISSED — no NPS contribution
            "hra":   120000,
            "other": 0,
        },
        "tds_deducted": 97500,
    }


def _parse_float(s: str) -> float:
    try:
        return float(str(s).replace(",", "").replace("Rs", "").replace("₹", "").strip())
    except ValueError:
        return 0.0