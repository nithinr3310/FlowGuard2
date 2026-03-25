"""
flowguard/api/whatsapp_webhook.py
──────────────────────────────────
WhatsApp bot webhook handler (Twilio Sandbox or official WhatsApp Business API).

HOW IT WORKS:
  1. User sends a WhatsApp message to your Twilio number.
  2. Twilio POSTs to this endpoint (POST /whatsapp).
  3. This handler detects the intent, calls the internal pipeline,
     and sends the narrated result back via Twilio.

SETUP:
  pip install twilio
  Set env vars:
    TWILIO_ACCOUNT_SID=ACxxxxxxxx
    TWILIO_AUTH_TOKEN=xxxxxxxx
    TWILIO_WHATSAPP_FROM=whatsapp:+14155238886  (Twilio sandbox number)

  In Twilio console → Messaging → WhatsApp Sandbox:
    Webhook URL: https://your-domain/whatsapp
    Method: POST

MULTI-TURN SESSION:
  Each user's phone number gets its own session dict (in-memory).
  A session stores the last ScoreRequest so the user can:
    - Ask "what if Kapoor pays 30k tomorrow?" without re-entering obligations.
    - Say "EMAIL 1" to draft a delay email for the first obligation.
    - Say "FULL" to get the complete web-channel narrative.

COMMANDS (WhatsApp):
  <any obligation text>    → run pipeline, send narrated result
  FULL                     → send full web-channel narrative of last result
  EMAIL <n>                → draft delay email for decision n
  WHATIF <amount> <when>   → scenario analysis
  RESET                    → clear session
  HELP                     → instructions
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Form, Request, Response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/whatsapp", tags=["WhatsApp Bot"])

# ── Session Store (abstracted for Redis migration) ─────────────────────

from abc import ABC, abstractmethod


class SessionStore(ABC):
    """Abstract session store. Swap InMemorySessionStore for RedisSessionStore in prod."""

    @abstractmethod
    def get(self, phone: str) -> dict: ...

    @abstractmethod
    def clear(self, phone: str) -> None: ...


class InMemorySessionStore(SessionStore):
    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def get(self, phone: str) -> dict:
        if phone not in self._store:
            self._store[phone] = {
                "last_score_request": None,
                "last_engine_result": None,
                "last_narrative":     None,
            }
        return self._store[phone]

    def clear(self, phone: str) -> None:
        self._store.pop(phone, None)


_session_store = InMemorySessionStore()

HELP_TEXT = (
    "*FlowGuard WhatsApp Bot* 📊\n\n"
    "Tell me your bills and cash balance, like:\n"
    "_gst 20k friday, rent 25k monday, cash 1 lakh_\n\n"
    "Commands:\n"
    "• *FULL* — full detailed report\n"
    "• *EMAIL 1* — draft delay email for bill #1\n"
    "• *WHATIF 30000 tomorrow* — scenario analysis\n"
    "• *RESET* — start over\n"
    "• *HELP* — show this message"
)


def _twilio_reply(text: str) -> Response:
    """Return a TwiML XML response that Twilio sends back to WhatsApp."""
    # Escape XML special chars minimally
    text_escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Message>{text_escaped}</Message>"
        "</Response>"
    )
    return Response(content=xml, media_type="application/xml")


@router.post("")
async def whatsapp_webhook(
    request: Request,
    From: str = Form(...),
    Body: str = Form(...),
):
    """
    Twilio WhatsApp webhook.

    IN : Twilio POST form — From (phone), Body (message text)
    OUT: TwiML XML response (Twilio sends it to WhatsApp)
    """
    from .main import _parse_raw_to_score_request
    from .scorer import run_engine
    from .parser import (
        classify_intent,
        draft_negotiation_email,
        extract_what_if_params,
        narrate_result,
        narrate_whatsapp_preview,
    )
    from .models import NLPParseRequest, CashPosition

    phone   = From.strip()
    message = Body.strip()
    session = _session_store.get(phone)

    logger.info("WhatsApp [%s]: %s", phone, message[:80])

    msg_upper = message.upper().strip()

    # ── RESET ──────────────────────────────────────────────────────────
    if msg_upper == "RESET":
        _session_store.clear(phone)
        return _twilio_reply("Session cleared. Send your bills to start fresh. 🔄")

    # ── HELP ───────────────────────────────────────────────────────────
    if msg_upper in ("HELP", "HI", "HELLO", "START"):
        return _twilio_reply(HELP_TEXT)

    # ── FULL — resend last result in web format ────────────────────────
    if msg_upper == "FULL":
        if not session.get("last_engine_result"):
            return _twilio_reply("No previous analysis. Send your bills first.")
        full_text = narrate_result(session["last_engine_result"], channel="web")
        # Truncate to Twilio 1600-char limit
        return _twilio_reply(full_text[:1580])

    # ── EMAIL <n> ──────────────────────────────────────────────────────
    if msg_upper.startswith("EMAIL"):
        if not session.get("last_engine_result"):
            return _twilio_reply("No previous analysis. Send your bills first.")
        parts = message.split()
        try:
            idx = int(parts[1]) - 1  # user uses 1-based
        except (IndexError, ValueError):
            idx = 0
        decisions = session["last_engine_result"].decisions
        if idx < 0 or idx >= len(decisions):
            return _twilio_reply(
                f"Invalid number. You have {len(decisions)} obligation(s). "
                f"Use EMAIL 1 through EMAIL {len(decisions)}."
            )
        email_body = draft_negotiation_email(
            decisions[idx],
            proposed_date=date.today() + timedelta(days=7),
        )
        reply = f"*Draft email for {decisions[idx].counterparty_name}:*\n\n{email_body}"
        return _twilio_reply(reply[:1580])

    # ── WHATIF ─────────────────────────────────────────────────────────
    if msg_upper.startswith("WHATIF") or "what if" in message.lower():
        if not session.get("last_score_request"):
            return _twilio_reply("No previous analysis. Send your bills first.")
        params = extract_what_if_params(message)
        if not params.get("inflow_amount"):
            return _twilio_reply(
                "Could not parse amount. Try: *WHATIF 30000 tomorrow*"
            )
        # Modify cash position
        base_req = session["last_score_request"]
        modified_cash = base_req.cash_position.model_copy(deep=True)
        day_key = str(params.get("inflow_day_offset", 1))
        modified_cash.expected_inflows[day_key] = (
            modified_cash.expected_inflows.get(day_key, 0.0)
            + params["inflow_amount"]
        )
        modified_result = run_engine(
            obligations=base_req.obligations,
            cash_position=modified_cash,
        )
        narrative = narrate_result(modified_result, channel="whatsapp")
        reply = (
            f"📊 *What-if: +₹{params['inflow_amount']:,.0f} in {day_key} day(s)*\n\n"
            + narrative
        )
        session["last_engine_result"] = modified_result
        return _twilio_reply(reply[:1580])

    # ── MAIN FLOW: parse obligations ───────────────────────────────────
    try:
        parse_req = NLPParseRequest(raw_text=message, language="en")
        score_req = _parse_raw_to_score_request(parse_req)
    except Exception as e:
        logger.warning("Parse error for [%s]: %s", phone, e)
        return _twilio_reply(
            "I couldn't understand that. Try:\n"
            "_gst 20k friday, rent 25k monday, cash 1 lakh_\n\n"
            "Or type *HELP*."
        )

    # Run engine
    result = run_engine(
        obligations=score_req.obligations,
        cash_position=score_req.cash_position,
    )

    # Store session
    session["last_score_request"] = score_req
    session["last_engine_result"] = result

    narrative = narrate_result(result, channel="whatsapp")
    session["last_narrative"] = narrative

    return _twilio_reply(narrative[:1580])
