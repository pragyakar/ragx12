"""RAG + LangChain based summarization core for ragX12 API.

This module exposes a single public function: ``generate_summary(x12_file: str)``
that produces a concise human-readable summary of an X12 healthcare EDI claim
using a Retrieval-Augmented Generation (RAG) workflow, PDO prompt engineering,
and LangSmith observability hooks.

DEPENDENCIES (install before use):
    pip install langchain langchain_community langchain-openai faiss-cpu python-dotenv openai

NOTES:
- Uses FAISS in-memory vector store populated from a mock knowledge base of
  X12 segment definitions and denial codes.
- Demonstrates PDO (Problem / Details / Objective) prompt construction.
- Includes a mock X12 parser tool and a retriever tool; both are orchestrated
  to assemble a final consolidated prompt for the LLM (ChatOpenAI placeholder).
- LangSmith environment variables are set with placeholders; replace with real
  values to enable tracing.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
import os
import json
import textwrap
import re  # For section parsing into structured JSON
from dataclasses import dataclass

# --- LangChain & related imports (with fallbacks) ---------------------------------
# Core LangChain components for LLMs, embeddings, vector stores, and agents.
try:  # New modular import style first
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # type: ignore
except ImportError:  # Fallback to legacy path for broader compatibility
    from langchain.chat_models import ChatOpenAI  # type: ignore
    from langchain.embeddings import OpenAIEmbeddings  # type: ignore

from langchain.vectorstores import FAISS  # FAISS in-memory similarity search
from langchain.schema import Document
try:  # New core messages import path
    from langchain_core.messages import HumanMessage
except ImportError:  # Legacy fallback
    from langchain.schema import HumanMessage  # type: ignore
from langchain.tools import tool  # To wrap parser as a LangChain tool
from langchain.tools.retriever import create_retriever_tool

# Optional (nice to have) for deterministic hashing of KB items or future caching
import hashlib

# ----------------------------------------------------------------------------------

# LANGSMITH / OBSERVABILITY PLACEHOLDER CONFIG -------------------------------------
# Replace the placeholder values below with real keys to enable LangSmith tracing.
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")  # set to "true" to enable
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ.setdefault("LANGCHAIN_API_KEY", "YOUR_LANGSMITH_API_KEY")  # <-- Replace

# OPENAI API KEY is REQUIRED (do NOT silently set a placeholder). If missing, summary generation will raise.
# Do not set a default here to avoid masking configuration errors.


# ------------------------------ MOCK KNOWLEDGE BASE --------------------------------
# Each entry: (identifier, natural language description)
_SEGMENT_DEFINITIONS: List[Tuple[str, str]] = [
    ("ST*837", "Transaction Set Header for Health Care Claim (837). Identifies the start of a health care claim submission."),
    ("NM1*IL", "Individual or Insured's Name (IL). Typically the subscriber / patient individual-level identification details."),
    ("NM1*82", "Rendering Provider Name (82). Identifies the provider who performed the service."),
    ("CLM*01", "Claim Information (CLM). Contains claim submitter identifier and high-level claim financial data."),
    ("HI*ABK", "Diagnosis code (principal). ABK indicates ICD-10 principal diagnosis code."),
    ("SV1*HC", "Professional service line (HC). Contains procedure code, charge amount, units."),
    ("REF*D9", "Claim Identifier for Transmission Intermediaries. Often a clearinghouse control reference."),
    ("DTP*472", "Service Date. Specifies the date(s) of service for the claim or service line."),
]

_DENIAL_CODES: List[Tuple[str, str]] = [
    ("CO-45", "Charges exceed the maximum allowable amount (Contractual Obligation)."),
    ("PR-1", "Deductible amount. Patient responsibility for deductible portion."),
    ("CO-97", "The benefit for this service is included in the payment/allowance for another service/procedure."),
]

# Core code/segment dictionary (concise meanings). In absence of true internet lookup, this
# serves as an internal curated reference. Extend as needed.
CODE_DEFINITIONS: Dict[str, str] = {
    # Segment + purpose
    **{seg: desc for seg, desc in _SEGMENT_DEFINITIONS},
    # Denial codes
    **{code: desc for code, desc in _DENIAL_CODES},
    # Common CAS group codes
    "CO": "Contractual Obligation adjustment (non-payable per contract).",
    "PR": "Patient Responsibility amount (owed by patient).",
    "OA": "Other Adjustment (miscellaneous).",
    "PI": "Payer Initiated reduction (policy/administrative).",
    # Other frequently referenced segments
    "CLP": "Claim Level Payment information (835 remittance).",
    "CAS": "Claims Adjustment Segment (adjustment group/reason codes and amounts).",
    "NM1": "Individual or organization name segment (entity identification).",
    "HI": "Health Care Diagnosis Code segment (diagnoses).",
    "SV1": "Professional Service line (procedure, charge, units).",
    "SV2": "Institutional Service line (revenue code, HCPCS).",
    "SV3": "Dental Service line (ADA code, charge).",
    "DTP": "Date or Time Period (service or event dates).",
    "REF": "Reference Identification (supplemental identifiers).",
    "AMT": "Monetary Amount segment.",
    "PLB": "Provider Level Balance (year-end or bulk adjustments).",
}


def _build_kb_documents() -> List[Document]:
    """Convert mock KB tuples into LangChain Document objects."""
    docs: List[Document] = []
    for code, desc in _SEGMENT_DEFINITIONS:
        docs.append(Document(page_content=f"Segment {code}: {desc}", metadata={"type": "segment", "code": code}))
    for code, desc in _DENIAL_CODES:
        docs.append(Document(page_content=f"Denial {code}: {desc}", metadata={"type": "denial", "code": code}))
    return docs


_VECTORSTORE = None  # Lazy-initialized FAISS store


def _get_vectorstore() -> FAISS:
    """Build (once) and return a FAISS vector store over the mock KB."""
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    # Validate OpenAI key BEFORE attempting to create embeddings
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key.strip() == "YOUR_OPENAI_API_KEY":
        raise ValueError("OPENAI_API_KEY environment variable is required for embeddings.")

    docs = _build_kb_documents()
    # Embeddings: Using OpenAI embeddings (requires OPENAI_API_KEY).
    embeddings = OpenAIEmbeddings()  # Will fail fast if key invalid
    _VECTORSTORE = FAISS.from_documents(docs, embeddings)
    return _VECTORSTORE


# ------------------------------ MOCK X12 PARSER TOOL --------------------------------
def _x12_parser(file_content: str) -> Dict[str, Any]:
    """Mock parser for X12 claim content.

    Splits content on tildes (~) or newlines, extracts the segment ID (token
    before first '*'), and groups segments.
    """
    raw = [seg.strip() for part in file_content.split("\n") for seg in part.split("~")]
    segments = [s for s in raw if s]
    parsed: Dict[str, List[str]] = {}
    for seg in segments:
        seg_id = seg.split('*', 1)[0]
        parsed.setdefault(seg_id, []).append(seg)
    return {"segments": parsed, "segment_count": len(segments)}


@tool("x12_parser", return_direct=False)
def x12_parser_tool(x12_text: str) -> str:
    """Parse X12 content and return a JSON summary of segments."""
    parsed = _x12_parser(x12_text)
    return json.dumps(parsed, indent=2)  # Agent-friendly text output


# Prepare retriever tool (lazy so embeddings only created if needed)
def _get_retriever_tool():
    vs = _get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 6})
    return create_retriever_tool(
        retriever,
        name="kb_lookup",
        description="Retrieve definitions for X12 segments & denial codes to aid summarization.",
    )


@dataclass
class _RAGContext:
    parsed: Dict[str, Any]
    retrieved_texts: List[str]


def _gather_context(x12_content: str) -> _RAGContext:
    """Parse file and retrieve related knowledge base context."""
    parsed = _x12_parser(x12_content)
    segment_ids = list(parsed.get("segments", {}).keys())
    queries = segment_ids.copy()
    for denial_code, _ in _DENIAL_CODES:
        if denial_code in x12_content:
            queries.append(denial_code)

    retriever_tool = _get_retriever_tool()
    retrieved_chunks: List[str] = []
    for q in queries:
        try:
            docs = retriever_tool.retriever.get_relevant_documents(q)  # type: ignore[attr-defined]
            for d in docs:
                retrieved_chunks.append(d.page_content)
        except Exception:  # pragma: no cover
            continue

    # Deduplicate
    seen = set()
    deduped: List[str] = []
    for t in retrieved_chunks:
        h = hashlib.sha1(t.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(t)

    return _RAGContext(parsed=parsed, retrieved_texts=deduped)


def _build_pdo_prompt(x12_content: str, ctx: _RAGContext) -> str:
    """Construct PDO (Problem / Details / Objective) formatted prompt."""
    # Attempt structured financial extraction early (esp. for 835 remittance)
    financial_struct = _extract_financials(x12_content)
    financial_json = json.dumps(financial_struct, indent=2)
    problem = (
        "PROBLEM:\nGenerate a detailed, human-readable summary of the provided X12 healthcare claim file."
    )

    parsed_segments = ctx.parsed.get("segments", {})
    parsed_preview_lines: List[str] = []
    for seg_id, seg_list in list(parsed_segments.items())[:25]:
        for seg in seg_list[:3]:
            parsed_preview_lines.append(seg)
    parsed_preview = "\n".join(parsed_preview_lines)

    retrieved_definitions = "\n".join(ctx.retrieved_texts[:40])

    details = textwrap.dedent(
        f"""DETAILS:
    Raw X12 (truncated to first 1500 chars):\n{x12_content[:1500]}\n\n
    Parsed Segment Examples (subset):\n{parsed_preview}\n\n
    Retrieved Knowledge Base Definitions & Denial Codes (subset):\n{retrieved_definitions}\n\n
    STRUCTURED FINANCIAL PARSED DATA (authoritative – DO NOT CONTRADICT):\n{financial_json}\n\n
    Instructions: Identify patient, provider, key claim/line items, financial amounts, and any denial/rejection indicators. Explain in plain English. Avoid raw EDI jargon unless necessary (include segment code in parentheses if helpful). Highlight anomalies or missing critical segments. If denial codes appear, explain their meaning and potential resolution steps succinctly.
    """
    ).strip()

    objective = (
        "OBJECTIVE:\nIf the provided X12 content appears MALFORMED or INCOMPLETE (e.g., missing ST/SE pairing, truncated mid-segment, contains mostly random text not following SEG*... pattern, or lacks any recognizable transaction set), respond with EXACTLY the single line: X12 file content is invalid"
        "\nOtherwise, return ONLY a human-readable summary starting with a single line for the transaction file type followed by the numbered sections below."
        "\nThe FIRST line (for valid files) MUST be exactly: File Type: <one of 837p, 837i, 837d, 837, 835, 277, 278, 270, 271, unknown>"
        "\nThen output the following numbered headings EACH on its own line, in order, with their content below them:"
        "\n1. Claim Overview"
        "\n2. Participants (Patient / Provider)"
        "\n3. Services & Charges"
        "\n4. Financial Summary"
        "\n5. Denials / Issues / Recommendations"
        "\n6. Key Codes Referenced"
    "\nSection 6 MUST list each referenced segment or denial/adjustment code followed by a hyphen and its concise description pulled verbatim (or near verbatim) from the provided knowledge base definitions when available."
    " If a code appears that is not in the knowledge base, provide a best-effort concise plain English description or write 'description unknown'."
    " Include denial codes (e.g., CO-45, PR-1, CO-97) with their meanings."
        "\nSTRICT FINANCIAL ACCURACY RULES (for 835):"
        " - Use STRUCTURED FINANCIAL PARSED DATA values exactly (do NOT recalc differently)."
        " - In CLP segments: CLP03=Billed, CLP04=Paid, CLP05=Patient Responsibility."
        " - Sum across all CLP segments: billed_total, paid_total, patient_responsibility_total."
        " - CAS segment amounts are adjustment amounts; sum as adjustments_total."
        " - Never claim a claim was 'paid in full' unless: paid_total == billed_total AND adjustments_total == 0 AND patient_responsibility_total == 0."
        " - If paid_total == 0 AND billed_total > 0 AND (adjustments_total > 0 OR patient_responsibility_total > 0), explicitly state it was NOT paid."
        " - If claim status code (CLP02) indicates denial (e.g., 2) reflect denial and cite relevant CAS reason codes."
        " - Allowed/Allowed Amount: Only compute if determinable; otherwise omit or state 'Not specified'. Do NOT fabricate an allowed amount."
        " - List adjustments grouped by CAS group and reason code with amounts."
        "\nRules: If a section has no data write 'Not specified'. Do NOT add extra sections or JSON."
        "\nIf unsure about file type, use 'unknown'."
        "\nWrite clearly for a non-technical healthcare operations stakeholder."
    )
    return f"{problem}\n\n{details}\n\n{objective}"


def _extract_financials(x12_content: str) -> Dict[str, Any]:
    """Extract core financial metrics from 835/claim-like segments.

    Returns dict with:
        billed_total, paid_total, patient_responsibility_total,
        adjustments_total, adjustments (list of {group, reason, amount}),
        claim_status_codes (set), clp_count
    Safe for non-835 (returns zeros).
    """
    billed = paid = patient_resp = 0.0
    adjustments: List[Dict[str, Any]] = []
    claim_status_codes = set()
    clp_count = 0
    # Simple segment scan
    for raw_seg in x12_content.split('~'):
        seg = raw_seg.strip()
        if not seg:
            continue
        parts = seg.split('*')
        tag = parts[0]
        if tag == 'CLP' and len(parts) >= 6:
            clp_count += 1
            # CLP02 claim status code
            claim_status_codes.add(parts[1]) if len(parts) > 1 else None
            # Monetary fields (defensive parsing)
            try:
                billed += float(parts[2]) if parts[2] else 0.0
            except ValueError:
                pass
            try:
                paid += float(parts[3]) if parts[3] else 0.0
            except ValueError:
                pass
            try:
                patient_resp += float(parts[4]) if parts[4] else 0.0
            except ValueError:
                pass
        elif tag == 'CAS' and len(parts) >= 4:
            group_code = parts[1]
            # Reason/amount triplets starting at index 2
            trip = parts[2:]
            i = 0
            while i + 1 < len(trip):
                reason = trip[i]
                try:
                    amount = float(trip[i+1])
                except ValueError:
                    amount = 0.0
                adjustments.append({"group": group_code, "reason": reason, "amount": amount})
                i += 3  # skip quantity if present
    adjustments_total = sum(a['amount'] for a in adjustments)
    return {
        "billed_total": round(billed, 2),
        "paid_total": round(paid, 2),
        "patient_responsibility_total": round(patient_resp, 2),
        "adjustments_total": round(adjustments_total, 2),
        "adjustments": adjustments,
        "claim_status_codes": sorted(claim_status_codes),
        "clp_count": clp_count,
    }

def _is_obviously_malformed(x12_content: str) -> bool:
    """Lightweight heuristic checks for clearly malformed / incomplete X12.

    Heuristics (any triggering returns True):
      - File shorter than minimal envelope threshold (< 20 chars)
      - Missing both 'ST*' and 'ISA' segments
      - Contains ST without matching SE count
      - Ends without segment terminator '~' (suggests truncation) and length > 80
      - More than 30% of non-whitespace chars are NOT in allowed charset (A-Z0-9*~:\n\r) -> noise
    """
    data = x12_content.strip()
    if len(data) < 20:
        return True
    upper = data.upper()
    if 'ST*' not in upper and not upper.startswith('ISA'):
        return True
    st_count = upper.count('ST*')
    se_count = upper.count('SE*')
    if st_count != se_count or st_count == 0:
        return True
    if not upper.rstrip().endswith('~') and len(upper) > 80:
        return True
    # Noise ratio
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*~:\n\r ")
    meaningful = [c for c in upper if not c.isspace()]
    if not meaningful:
        return True
    bad = sum(1 for c in meaningful if c not in allowed)
    if bad / max(1, len(meaningful)) > 0.30:
        return True
    return False

def _collect_codes(x12_content: str, structured: Dict[str, Any]) -> List[str]:
    """Collect unique codes referenced in the file & structured summary.

    Includes:
      - CAS group codes (CO, PR, OA, PI)
      - Denial/adjustment codes like CO-45, PR-1 (pattern <AA>-<digits>)
      - Segment+qualifier patterns like ST*837, NM1*IL, NM1*82
      - Raw segment IDs (ST, CLP, CAS, NM1, HI, SV1, SV2, SV3, DTP, REF, AMT, PLB)
    """
    codes: set[str] = set()
    text = x12_content.upper()
    # Denial patterns (e.g. CO-45)
    for m in re.findall(r"\b([A-Z]{2})-(\d{1,3})\b", text):
        codes.add(f"{m[0]}-{m[1]}")
    # CAS group codes present anywhere
    for grp in ["CO", "PR", "OA", "PI"]:
        if grp + '-' in text or f"CAS*{grp}" in text:
            codes.add(grp)
    # Segment with qualifier (e.g., NM1*IL, NM1*82, ST*837)
    for m in re.findall(r"\b([A-Z0-9]{2,4}\*[A-Z0-9]{2,4})\b", text):
        # Only keep those we have definitions for (avoid noise) or core ST*837 etc
        if any(m.startswith(prefix) for prefix in ["ST*", "NM1*", "SV1*", "SV2*", "SV3*", "HI*", "CLP*", "CAS*", "REF*", "DTP*", "AMT*", "PLB*"]):
            codes.add(m)
    # Raw segment tags
    for tag in ["ST","CLP","CAS","NM1","HI","SV1","SV2","SV3","DTP","REF","AMT","PLB"]:
        if tag + '*' in text:
            codes.add(tag)
    # From structured key_codes_list if present
    for c in structured.get("key_codes_list", []) or []:
        codes.add(c.upper())
    # Sort for determinism
    return sorted(codes)


def _build_code_meanings(codes: List[str]) -> Dict[str, str]:
    """Map codes to concise meanings using CODE_DEFINITIONS with graceful fallbacks."""
    out: Dict[str, str] = {}
    for code in codes:
        # Exact match
        if code in CODE_DEFINITIONS:
            out[code] = CODE_DEFINITIONS[code]
            continue
        # Segment with qualifier (e.g., NM1*IL) -> attempt full then base segment
        if '*' in code:
            base = code.split('*', 1)[0]
            if code in CODE_DEFINITIONS:
                out[code] = CODE_DEFINITIONS[code]
                continue
            if base in CODE_DEFINITIONS:
                out[code] = CODE_DEFINITIONS[base]
                continue
        # Denial code maybe not in dict -> attempt group meaning
        if '-' in code:
            grp = code.split('-',1)[0]
            if grp in CODE_DEFINITIONS:
                out[code] = CODE_DEFINITIONS[grp] + " (specific reason code)"
                continue
        out[code] = "description unknown"
    return out


def _structure_summary(summary_text: str) -> Dict[str, Any]:
    """Parse the narrative summary into structured sections.

    Relies on the numbered headings enforced in the PDO objective prompt.
    Falls back gracefully if patterns not found.
    """
    # Regex to capture numbered section heading + body until next numbered heading
    # Capture numbered sections
    pattern = re.compile(r"(?:^|\n)(\d\.\s*[^\n]+)\n(.*?)(?=\n\d\.\s*[^\n]+|\Z)", re.DOTALL)
    found = pattern.findall(summary_text)
    heading_map = {}
    for heading, body in found:
        heading_map[heading.lower()] = body.strip()

    # Extract File Type line (must be first line per prompt). Accept case variations and tolerate leading spaces.
    file_type_match = re.search(r"^\s*File Type:\s*(.+)$", summary_text, re.IGNORECASE | re.MULTILINE)
    file_type_val = file_type_match.group(1).strip() if file_type_match else ""
    # Normalize file type token (strip trailing punctuation and lowercase)
    if file_type_val:
        file_type_val = file_type_val.split()[0].lower().rstrip('.,;:')
    allowed = {"837p","837i","837d","837","835","277","278","270","271","unknown"}
    if file_type_val not in allowed:
        file_type_val = "unknown"

    def pick(index_prefix: str, keyword: str) -> str:
        for h, body in heading_map.items():
            if h.startswith(index_prefix) or keyword in h:
                return body
        return ""

    structured: Dict[str, Any] = {
        "file_type": file_type_val,
        "claim_overview": pick("1.", "claim overview"),
        "participants": pick("2.", "participants"),
        "services_charges": pick("3.", "services"),
        "financial_summary": pick("4.", "financial"),
        "denials_issues_recommendations": pick("5.", "denials"),
        "key_codes_referenced": pick("6.", "key codes"),
    }

    # Heuristic enrichment for any empty sections
    if not structured["claim_overview"]:
        structured["claim_overview"] = " ".join(summary_text.split("\n")[:3]).strip()[:800] or "Not specified"
    if not structured["participants"]:
        participants_candidates = re.findall(r"NM1\*[A-Z0-9*:-]+", summary_text)
        structured["participants"] = ", ".join(participants_candidates[:5]) or "Not specified"
    if not structured["services_charges"]:
        svc_lines = [l for l in summary_text.splitlines() if any(k in l.lower() for k in ["procedure", "service", "charge", "sv1", "units"])]
        structured["services_charges"] = " ".join(svc_lines[:5]).strip() or "Not specified"
    if not structured["financial_summary"]:
        money_lines = re.findall(r"\b\$?\d+[,.]?\d*\b", summary_text)
        structured["financial_summary"] = ("Amounts mentioned: " + ", ".join(money_lines[:10])) if money_lines else "Not specified"
    if not structured["denials_issues_recommendations"]:
        denial_hits = [l for l in summary_text.splitlines() if any(code.lower() in l.lower() for code, _ in _DENIAL_CODES)]
        structured["denials_issues_recommendations"] = " ".join(denial_hits[:5]).strip() or "Not specified"
    if not structured["key_codes_referenced"]:
        # Collect codes from earlier segments
        potential = re.findall(r"\b[A-Z]{2,4}-?\d{0,4}\b", summary_text)
        structured["key_codes_referenced"] = ", ".join(sorted(set(potential))[:15]) or "Not specified"

    # Extract potential code tokens (simple heuristic)
    if structured.get("key_codes_referenced"):
        codes = re.findall(r"\b[A-Z0-9]{2,7}(?:-\d+)?\b", structured["key_codes_referenced"])
        structured["key_codes_list"] = sorted({c for c in codes if not c.isdigit()})

    return structured


def _detect_file_type(x12_content: str) -> str:
    """Heuristically detect the X12 transaction file type.

    Rules:
    - Look for first ST* segment token (e.g., ST*837, ST*835, ST*277, ST*278)
    - For 837 variants attempt to distinguish professional/institutional/dental via
      presence of service line segment patterns:
        * SV1 -> professional (837p)
        * SV2 -> institutional (837i)
        * SV3 -> dental (837d)
    - Fallback to base code or 'unknown'.
    """
    # Normalize spacing
    sample = x12_content.upper()
    # Find ST*transactionSetID
    file_type = None
    for seg in sample.replace('\n', '~').split('~'):
        if seg.startswith('ST*'):
            parts = seg.split('*')
            if len(parts) > 1:
                ts_id = parts[1]
                if ts_id in {"835", "837", "277", "278", "270", "271"}:
                    file_type = ts_id
                break
    if file_type == '837':
        # Distinguish variant
        if 'SV1*' in sample:
            return '837p'
        if 'SV2*' in sample:
            return '837i'
        if 'SV3*' in sample:
            return '837d'
        return '837'
    if file_type:
        return file_type
    if '835*' in sample or 'ST*835' in sample:
        return '835'
    return 'unknown'


def generate_summary(x12_content: str) -> Dict[str, Any]:
    """Generate a RAG-enhanced summary, structured data, and possible actions.

    possible_actions: LLM-derived recommended follow-up steps based on detected denials/issues.
    """
    if not x12_content or '*' not in x12_content or '~' not in x12_content:
        raise ValueError("File content is invalid!")
    if not any(x12_content.lstrip().startswith(prefix) for prefix in ("ISA", "ST")):
        raise ValueError("File content is invalid!")
    if _is_obviously_malformed(x12_content):
        raise ValueError("X12 file content is invalid")

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key.strip() == "YOUR_OPENAI_API_KEY":
        raise ValueError("OPENAI_API_KEY is missing. Set it in the environment or .env file.")

    ctx = _gather_context(x12_content)
    prompt = _build_pdo_prompt(x12_content, ctx)
    if not prompt.strip():
        raise ValueError("Internal error: failed to construct prompt for summarization.")

    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-nano")
    llm = ChatOpenAI(model=model_name, temperature=0)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
    except TypeError:
        response = llm.invoke(prompt)
    if response is None:
        raise ValueError("LLM returned no content.")
    summary_text = getattr(response, "content", str(response))
    if not summary_text.strip():
        raise ValueError("LLM produced empty summary output.")
    if summary_text.strip() == "X12 file content is invalid":
        raise ValueError("X12 file content is invalid")

    structured = _structure_summary(summary_text)
    ft = structured.get("file_type", "").lower()
    if not ft or ft == "unknown":
        structured["file_type"] = _detect_file_type(x12_content)
    actions = _generate_possible_actions(summary_text, structured)
    # Collect code meanings (post-summary to leverage extracted list)
    code_list = _collect_codes(x12_content, structured)
    code_meanings = _build_code_meanings(code_list)
    return {"summary": summary_text, "data": structured, "possible_actions": actions, "code_meanings": code_meanings}


def _generate_possible_actions(summary_text: str, structured: Dict[str, Any]) -> List[str]:
    """Use LLM to derive possible actions. Fall back to heuristics if call fails.

    Strategy:
      - Build concise action prompt referencing denials/issues section & codes.
      - Request bullet list (max 6) or explicit 'No action required at the moment'.
    """
    denials_section = structured.get("denials_issues_recommendations") or structured.get("denials_issues_recommendations", "")
    codes_line = structured.get("key_codes_referenced", "")
    base_context = f"Summary Section (Denials/Issues):\n{denials_section}\n\nCodes: {codes_line}\n"[:4000]
    action_prompt = (
        "ROLE: Senior medical claims resolution analyst."
        " TASK: Produce concrete, operational next-step actions based ONLY on the denial/issues text and codes below."
        " If there are no denial, rejection, adjustment, or follow-up indicators, respond with exactly: No action required at the moment"
        " REQUIRED FORMAT: Each action on its own line starting with '- '. 1 to 6 lines maximum."
        " STYLE: Imperative, specific, concise (<18 words), reference code(s) in parentheses when tied to a code."
        " CONTENT RULES:"
        " - Do not invent data not implied by text."
        " - If a code maps to a common resolution (e.g., CO-45 contractual write-off) state that explicitly."
        " - Group similar remediation steps only once."
        " - Prefer verbs: Verify, Resubmit, Contact, Obtain, Document, Adjust."
        " - Never include explanations beyond the action itself."
        f"\n\nSOURCE:\n{base_context}\nEND SOURCE\n"
    )
    try:
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-nano")
        llm = ChatOpenAI(model=model_name, temperature=0)
        resp = llm.invoke(action_prompt)
        content = getattr(resp, "content", str(resp)) if resp else ""
    except Exception:
        content = ""
    if not content:
        # Heuristic fallback
        if any(k in (denials_section or '').lower() for k in ["denial", "reject", "co-", "pr-", "action", "follow"]):
            return ["Review denial codes and verify contractual adjustments.", "Contact payer if denial rationale unclear.", "Prepare corrected claim or supporting documentation."]
        return ["No action required at the moment"]

    normalized = content.strip()
    if normalized.lower().startswith("no action required"):
        return ["No action required at the moment"]
    lines = [l.strip() for l in normalized.splitlines() if l.strip()]
    actions: List[str] = []
    for l in lines:
        if l.lower().startswith("no action required"):
            return ["No action required at the moment"]
        if l.startswith('-'):
            l = l[1:].strip()
        if l and len(actions) < 6:
            # Compress whitespace and enforce length guideline
            compact = re.sub(r"\s+", " ", l).strip()
            if len(compact) > 120:
                compact = compact[:117].rstrip() + '...'
            actions.append(compact)
    if not actions:
        return ["No action required at the moment"]
    return actions


if __name__ == "__main__":
    # Example usage with a mock minimal X12 claim snippet.
    mock_x12 = (
        "ST*837*0001~NM1*IL*1*DOE*JOHN****MI*123456789~NM1*82*1*SMITH*ALICE****XX*9876543210~"
        "CLM*01*500***11:B:1*Y*A*Y*Y~HI*ABK:K12345~SV1*HC:99213*150*UN*1***1:2:3~DTP*472*D8*20250115~"
        "REF*D9*TRANS123~AMT*B6*400~AMT*F5*100~PLB*12345*20251231*CO-45*50~"
    )
    print("--- GENERATED SUMMARY ---")
    result = generate_summary(mock_x12)
    print(result["summary"])
    print("\n--- STRUCTURED DATA ---")
    print(json.dumps(result["data"], indent=2))
    print("\n--- CODE MEANINGS ---")
    print(json.dumps(result.get("code_meanings"), indent=2))
