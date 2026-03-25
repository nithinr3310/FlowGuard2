"""
flowguard/groq_client.py
─────────────────────────
LLM LAYER — Groq API integration for FlowGuard.

This module is the ONLY place where LLM calls happen.
The deterministic engine (scorer.py) is NEVER touched by this module.

Model Routing:
  • llama3-8b-8192        → user input parsing (text → JSON)
  • mixtral-8x7b-32768    → JSON correction fallback
  • llama-3.3-70b-versatile → COT narration + email drafts

STRICT RULES:
  1. NEVER compute scores — only format/explain them
  2. Use temperature=0 for deterministic LLM outputs
  3. Validate all Groq outputs with Pydantic before use
  4. Graceful fallback to regex/template if Groq unavailable
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# GROQ CLIENT INIT
# ─────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed. Set GROQ_API_KEY in environment manually.")

_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
_GROQ_AVAILABLE = False
_client = None

if _GROQ_API_KEY and _GROQ_API_KEY != "gsk_PLACEHOLDER":
    try:
        from groq import Groq
        _client = Groq(api_key=_GROQ_API_KEY)
        _GROQ_AVAILABLE = True
        logger.info("Groq client initialised successfully.")
    except ImportError:
        logger.warning("groq package not installed. pip install groq")
    except Exception as e:
        logger.warning("Groq client init failed: %s", e)
else:
    logger.info("Groq API key not set. LLM features disabled — using regex/template fallback.")


# ─────────────────────────────────────────────
# MODEL CONSTANTS
# ─────────────────────────────────────────────

MODEL_PARSE    = "llama3-8b-8192"           # Fast, cheap — input parsing
MODEL_FIXJSON  = "mixtral-8x7b-32768"       # Good at structured correction
MODEL_NARRATE  = "llama-3.3-70b-versatile"  # Best quality — narration + email
_MODEL         = MODEL_PARSE                 # Alias used by file_ingest.py


# ─────────────────────────────────────────────
# HELPER: safe Groq call with retry
# ─────────────────────────────────────────────

def _groq_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0,
    retries: int = 2,
) -> Optional[str]:
    """Make a Groq chat completion call with retry.
    Returns the response text, or None if Groq is unavailable/fails.
    """
    if not _GROQ_AVAILABLE or _client is None:
        return None

    for attempt in range(retries + 1):
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
        except Exception as e:
            logger.warning("Groq call failed (attempt %d/%d, model=%s): %s",
                          attempt + 1, retries + 1, model, e)
    return None


def _groq_chat_text(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0,
) -> Optional[str]:
    """Groq call that returns plain text (no JSON mode)."""
    if not _GROQ_AVAILABLE or _client is None:
        return None
    try:
        response = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content.strip() if content else None
    except Exception as e:
        logger.warning("Groq text call failed (model=%s): %s", model, e)
        return None


# ─────────────────────────────────────────────
# 1. INPUT PARSING — llama3-8b
# ─────────────────────────────────────────────

_PARSE_SYSTEM = """
You are FlowGuard's intelligent financial assistant for Indian MSMEs (Micro, Small, Medium Enterprises).
You support Tamil and English ONLY. Detect the user's language and reply in the same language.

━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — CLASSIFY INTENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Classify the user's message into exactly one intent:

• INGEST — User is reporting cash they HAVE or obligations they OWE.
  Signals: amounts with due dates, vendors, GST/TDS/EMI/rent/salary/supplier payments.
  Examples:
    "I have 2L cash. GST 40k by 20th, supplier 80k thursday"
    "கையில் 1.5L இருக்கு. வாடகை 25k, சப்ளையர் 60k"
    "received 50k from customer via UPI today"

• STATUS — User wants advice, a summary, or recommendations.
  Signals: questions about what to do, priorities, cash situation, payment order.
  Examples:
    "what should I pay first?", "how is my cash flow?", "show current status"
    "இப்போது என் நிலை என்ன?"

• FILTER — User wants to query or search existing records.
  Signals: "show me", "list all", "filter by", "find", "payments to X", "transactions this week".
  Examples:
    "show all payments to Sharma Papers"
    "list UPI transactions this month"
    "GST payments in March"
    "Sharma papers க்கு எத்தனை payments போனது?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — AI NAME NORMALISATION (CRITICAL for deduplication)
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before extracting any counterparty name, normalise it:
- Correct likely typos ("shaarma" → "Sharma", "sharmapapers" → "Sharma Papers")
- Convert to Title Case ("sharma papers" → "Sharma Papers")
- Remove accidental punctuation unless it's part of a brand name
- For well-known entities (GST, HDFC, ICICI, TDS, PF, ESI): use their exact standard name
- If unsure, make your best reasonable guess
This is critical — the database uses a SHA-256 fingerprint of the name. Normalisation ensures
  "shaarma papers" and "Sharma Papers" map to the SAME record, preventing duplicate obligations.

━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — EXTRACT DATA (based on intent)
━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ── If INGEST ──
  Extract ALL obligations and/or transactions.

  INDIAN CONTEXT — Obligation categories:
  • GST, TDS, PF, ESI, income tax, advance tax → STATUTORY
    - GST due monthly on 20th, Rs.100/day penalty + 18% p.a. interest. FIXED. Never defer.
    - TDS due 7th of next month. 1.5% per month penalty.
    - PF/ESI due 15th of month. Cannot defer.
  • EMI, bank loan, NBFC → SECURED_LOAN
    - NPA classification after 90 days overdue. Penalty 2-3% monthly.
  • Salary, wages, staff, employee → SALARY
    - Due 1st-7th of month. Labour Court complaint risk. Rs.25,000 penalty.
  • Rent, lease, office → RENT
    - Usually NEGOTIABLE unless eviction notice issued.
  • Electricity, power, water, internet, wifi → UTILITY
    - DEFERRABLE ~7-15 days. Disconnection then reconnection fee applies.
  • Supplier, vendor, raw material, purchase, invoice → TRADE_PAYABLE
    - Usually DEFERRABLE. Relationship risk if delayed repeatedly.
  • Anything else → OTHER

  FLEXIBILITY RULES:
  • STATUTORY → FIXED
  • SECURED_LOAN, SALARY, RENT → NEGOTIABLE
  • UTILITY, TRADE_PAYABLE, OTHER → DEFERRABLE

  DEFAULT DUE DATES (if not mentioned):
  • STATUTORY → next 20th of month
  • SALARY → 5 days from today
  • All others → 7 days from today

  INDIAN NUMBER FORMATS (handle ALL of these):
  • 2L = 2 lakh = 200000
  • 1.5L = 150000
  • 70k / 70K = 70000
  • 2cr = 20000000
  • Rs.20000 / ₹20000 = 20000
  • "twenty thousand" = 20000
  • "do lakh" / "irandu latcham" = 200000

  ── If FILTER ──
  Extract search parameters as filter_query. Supported filters:
  • counterparty_name — vendor/supplier/party name (normalised)
  • category — STATUTORY, TRADE_PAYABLE, etc.
  • direction — IN or OUT
  • medium — UPI, BANK_CHEQUE, RECEIPT, LIQUID_CASH, BANK_TRANSFER, ONLINE, DEMAND_DRAFT
  • date_from / date_to — ISO date strings YYYY-MM-DD

  ── If STATUS ──
  No data extraction needed. Just confirm intent.

━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — GENERATE BOT REPLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generate a short, natural language reply (in the SAME language as the user's message — Tamil or English):

• If INGEST (new data):
  English: "✅ Got it! Recorded [N] obligation(s). Sharma Papers ₹50,000 due 28 Mar → logged."
  Tamil: "✅ சரி! [N] கடமை(கள்) பதிவு செய்யப்பட்டது. Sharma Papers ₹50,000 → சேமிக்கப்பட்டது."

• If INGEST (duplicate detected, same ref_id already exists):
  English: "⚠️ Duplicate detected. This transaction (Sharma Papers ₹50,000 on 28 Mar) was already recorded. Skipped."
  Tamil: "⚠️ இந்த பரிவர்த்தனை ஏற்கனவே பதிவு செய்யப்பட்டது. தவிர்க்கப்பட்டது."

• If STATUS:
  English: "Sure! Analysing your current cash flow…"
  Tamil: "சரி! உங்கள் தற்போதைய நிதி நிலையை ஆராய்கிறேன்…"

• If FILTER:
  English: "Fetching matching records…"
  Tamil: "பொருந்தும் பதிவுகளை தேடுகிறேன்…"

━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON — no markdown, no explanation:

{
  "intent": "INGEST|STATUS|FILTER",
  "bot_reply": "short natural language reply in Tamil or English",
  "cash_balance_inr": <number or 0>,
  "obligations": [
    {
      "counterparty_name": "Sharma Papers",
      "description": "short phrase",
      "amount_inr": <number — no commas, no currency symbol>,
      "category": "STATUTORY|SECURED_LOAN|SALARY|RENT|UTILITY|TRADE_PAYABLE|OTHER",
      "due_date": "YYYY-MM-DD",
      "flexibility": "FIXED|NEGOTIABLE|DEFERRABLE"
    }
  ],
  "filter_query": {
    "counterparty_name": null,
    "category": null,
    "direction": null,
    "medium": null,
    "date_from": null,
    "date_to": null
  }
}

RULES:
- obligations → only populate for INGEST intent
- filter_query → only populate for FILTER intent
- obligations and filter_query → EMPTY for STATUS intent
- cash_balance_inr → what the user HAS (not owes). 0 if not mentioned.
- Amount: always a plain number. NEVER commas, currency symbols, or strings.
- NEVER invent obligations not mentioned by the user.
- NEVER compute scores — your job is EXTRACTION ONLY.

FEW-SHOT EXAMPLES:

Example A (INGEST, English):
User: "cash 2L, gst 40k by 20th and rent 30k end of month"
Output: {"intent":"INGEST","bot_reply":"✅ Got it! Recorded 2 obligations — GST ₹40,000 due 20 Mar and Landlord ₹30,000 due 31 Mar.","cash_balance_inr":200000,"obligations":[{"counterparty_name":"GST","description":"Monthly GST filing","amount_inr":40000,"category":"STATUTORY","due_date":"2026-03-20","flexibility":"FIXED"},{"counterparty_name":"Landlord","description":"Office rent","amount_inr":30000,"category":"RENT","due_date":"2026-03-31","flexibility":"NEGOTIABLE"}],"filter_query":{}}

Example B (INGEST, Tamil):
User: "கையில் 1.5 லட்சம் இருக்கு. சப்ளையர்க்கு 80k கொடுக்கணும், ஸ்டாஃப் சம்பளம் 60k"
Output: {"intent":"INGEST","bot_reply":"✅ சரி! 2 கடமைகள் பதிவு செய்யப்பட்டது — Supplier ₹80,000 மற்றும் Staff ₹60,000.","cash_balance_inr":150000,"obligations":[{"counterparty_name":"Supplier","description":"Supplier payment","amount_inr":80000,"category":"TRADE_PAYABLE","due_date":"2026-04-02","flexibility":"DEFERRABLE"},{"counterparty_name":"Staff","description":"Monthly salary","amount_inr":60000,"category":"SALARY","due_date":"2026-03-31","flexibility":"NEGOTIABLE"}],"filter_query":{}}

Example C (STATUS):
User: "what should I pay first?"
Output: {"intent":"STATUS","bot_reply":"Sure! Analysing your current cash flow…","cash_balance_inr":0,"obligations":[],"filter_query":{}}

Example D (FILTER):
User: "show all payments to shaarma papers this month"
Output: {"intent":"FILTER","bot_reply":"Fetching matching records…","cash_balance_inr":0,"obligations":[],"filter_query":{"counterparty_name":"Sharma Papers","date_from":"2026-03-01","date_to":"2026-03-31"}}

Example E (INGEST — Name normalisation):
User: "I paid shaarmapapers 50000 by cheque"
Output: {"intent":"INGEST","bot_reply":"✅ Got it! Recorded payment of ₹50,000 to Sharma Papers via Bank Cheque.","cash_balance_inr":0,"obligations":[{"counterparty_name":"Sharma Papers","description":"Cheque payment","amount_inr":50000,"category":"TRADE_PAYABLE","due_date":"2026-03-26","flexibility":"DEFERRABLE"}],"filter_query":{}}
"""


def groq_parse_input(
    raw_text: str,
    reference_date: Optional[date] = None,
) -> Optional[dict]:
    """Use Groq llama3-8b to extract intent + obligations + filter_query + bot_reply.
    Returns the full parsed dict with keys:
      intent, bot_reply, cash_balance_inr, obligations, filter_query.
    Returns None if unavailable or failed.
    """
    ref = reference_date or date.today()
    prompt = f"Today's date: {ref.isoformat()}\n\nUser message:\n{raw_text}"

    result = _groq_chat(MODEL_PARSE, _PARSE_SYSTEM, prompt, max_tokens=1500)
    if result is None:
        return None

    try:
        parsed = json.loads(result)
        # Ensure required top-level keys exist
        parsed.setdefault("intent", "STATUS")
        parsed.setdefault("bot_reply", "")
        parsed.setdefault("cash_balance_inr", 0)
        parsed.setdefault("obligations", [])
        parsed.setdefault("filter_query", {})

        if not isinstance(parsed["obligations"], list):
            parsed["obligations"] = []

        return parsed
    except json.JSONDecodeError as e:
        logger.warning("Groq parse returned invalid JSON: %s", e)
        return groq_fix_json(raw_text, result, str(e), reference_date=ref)



# ─────────────────────────────────────────────
# 2. JSON CORRECTION FALLBACK — mixtral
# ─────────────────────────────────────────────

_FIXJSON_SYSTEM = """You are a JSON repair specialist.
The user tried to extract financial data but the output was invalid JSON.
Fix the JSON and return ONLY the corrected, valid JSON.

The expected schema is:
{
  "obligations": [{"counterparty_name": str, "description": str, "amount_inr": number, "category": str, "due_date": "YYYY-MM-DD", "flexibility": str}],
  "cash_balance_inr": number
}

Return ONLY valid JSON. No explanations.
"""


def groq_fix_json(
    original_text: str,
    broken_json: str,
    error_msg: str,
    reference_date: Optional[date] = None,
) -> Optional[dict]:
    """Use Groq mixtral to fix broken JSON from a failed parse.
    Returns corrected dict or None.
    """
    ref = reference_date or date.today()
    prompt = (
        f"Original user input: {original_text}\n\n"
        f"Today's date: {ref.isoformat()}\n\n"
        f"Broken output:\n{broken_json}\n\n"
        f"Error: {error_msg}\n\n"
        f"Please return the corrected JSON."
    )

    result = _groq_chat(MODEL_FIXJSON, _FIXJSON_SYSTEM, prompt, max_tokens=1024)
    if result is None:
        return None

    try:
        parsed = json.loads(result)
        if "obligations" in parsed and "cash_balance_inr" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    logger.warning("Groq JSON fix also failed — falling back to regex parser")
    return None


# ─────────────────────────────────────────────
# 3. COT NARRATION — llama-3.3-70b
# ─────────────────────────────────────────────

_COT_SYSTEM = """You are a senior CFO advisor explaining machine-computed payment decisions to an Indian MSME owner.
You receive structured facts — the math is already done. Your job is to explain it naturally.

INDIAN REGULATORY CONTEXT (use this for accurate explanations):
- GST/TDS/PF: Missing payment = 18% annual interest + Rs.100/day + possible prosecution. Cannot defer.
- EMI/Bank loan: Overdue = NPA classification in 90 days, credit score harm, asset seizure risk.
- Salary: Delay = Labour Court complaint under Payment of Wages Act. Rs.25,000 max penalty.
- Rent: Overdue = eviction notice, forfeiture of security deposit.
- Supplier: Supply stop, credit line revoked. No legal penalty but operational shutdown risk.
- Utility: Disconnection after 15-30 days. Reconnection fee + downtime cost.

ACTION MEANING:
- PAY: Highest urgency — settle immediately from available cash.
- DEFER: Lower risk — can safely push without major consequence right now.
- NEGOTIATE: Call the party, propose a revised date. Relationship score supports this.
- ESCALATE: Cash critically short, legal deadline imminent — seek emergency credit NOW.

Return ONLY this JSON (no markdown, no explanation):
{
  "cot_reason": "1-2 sentences: why this obligation has this priority, citing specific amount, due date, and risk",
  "cot_tradeoff": "1 sentence: exact daily cost or operational risk if deferred even 1 day",
  "cot_downstream": "1 sentence: how settling/deferring this affects the remaining cash and other obligations"
}

Rules:
- Write in plain, direct Indian business English
- Always mention the specific amount in Rs. and the due date
- Quote the daily penalty amount if given in the facts
- NEVER change the action (PAY/DEFER/etc.) — only explain it
- NEVER invent numbers not present in the input facts
- Max 2 sentences per field
"""


def groq_rewrite_cot(decision_facts: dict) -> Optional[dict]:
    """Use Groq llama-3.3-70b to rewrite template COT into natural language.
    
    Args:
        decision_facts: Dict with keys like counterparty_name, amount_inr,
                       consequence_score, action, cot_reason, cot_tradeoff, 
                       cot_downstream, penalty_per_day_inr, due_date, score_band
    
    Returns:
        Dict with rewritten cot_reason, cot_tradeoff, cot_downstream.
        None if Groq unavailable.
    """
    prompt = json.dumps(decision_facts, default=str, indent=2)

    result = _groq_chat(MODEL_NARRATE, _COT_SYSTEM, prompt, max_tokens=512)
    if result is None:
        return None

    try:
        parsed = json.loads(result)
        required = {"cot_reason", "cot_tradeoff", "cot_downstream"}
        if required.issubset(parsed.keys()):
            return parsed
    except json.JSONDecodeError:
        pass

    logger.warning("Groq COT rewrite failed — using template fallback")
    return None


# ─────────────────────────────────────────────
# 4. EMAIL DRAFTING — llama-3.3-70b
# ─────────────────────────────────────────────

_EMAIL_SYSTEM = """You are a professional email writer for Indian businesses.
Draft a payment delay notification email based on the given facts.

The tone MUST match the tone_tag provided:
- WARM_APOLOGETIC: Friendly, relationship-focused, personal
- PROFESSIONAL_NEUTRAL: Formal but respectful, business-like
- FIRM_BRIEF: Short, factual, no fluff

Return JSON:
{
  "subject": "Email subject line",
  "body": "Full email body",
  "tone": "WARM_APOLOGETIC|PROFESSIONAL_NEUTRAL|FIRM_BRIEF"
}

Rules:
- Use Indian business conventions (Dear Sir/Madam for formal, Dear <name> for warm)
- Reference specific amounts in ₹
- Include the proposed reschedule date if given
- Keep it under 200 words
- Do NOT add your own commentary outside the email
"""


def groq_draft_email(
    counterparty_name: str,
    amount_inr: float,
    due_date: date,
    tone_tag: str,
    sender_name: str = "The Management",
    proposed_date: Optional[date] = None,
    reason: str = "temporary cash flow constraint",
) -> Optional[dict]:
    """Use Groq llama-3.3-70b to draft a negotiation email.
    Returns dict with subject, body, tone or None.
    """
    facts = {
        "counterparty_name": counterparty_name,
        "amount_inr": amount_inr,
        "due_date": due_date.isoformat(),
        "tone_tag": tone_tag,
        "sender_name": sender_name,
        "proposed_date": proposed_date.isoformat() if proposed_date else None,
        "reason": reason,
    }
    prompt = json.dumps(facts, indent=2)

    result = _groq_chat(MODEL_NARRATE, _EMAIL_SYSTEM, prompt, max_tokens=1024)
    if result is None:
        return None

    try:
        parsed = json.loads(result)
        if "subject" in parsed and "body" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    logger.warning("Groq email draft failed — using template fallback")
    return None


# ─────────────────────────────────────────────
# PUBLIC STATUS
# ─────────────────────────────────────────────

def is_groq_available() -> bool:
    """Check if Groq is configured and ready."""
    return _GROQ_AVAILABLE


def get_groq_status() -> dict:
    """Return status info for the /health endpoint."""
    return {
        "groq_available": _GROQ_AVAILABLE,
        "parse_model": MODEL_PARSE,
        "fixjson_model": MODEL_FIXJSON,
        "narrate_model": MODEL_NARRATE,
    }
