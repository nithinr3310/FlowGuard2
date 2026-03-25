# FlowGuard — Obligation Priority Engine

**Deterministic cash flow prioritisation for Indian MSMEs.**

> "FlowGuard doesn't show you your cash problem. It solves it — with math, not guesswork."

---

## Architecture

```
WhatsApp User
     │
     │  "gst 20k friday, rent 25k monday, cash 1 lakh"
     ▼
Twilio Webhook  (/whatsapp)
     │
     ▼
POST /pipeline  ◄──── ONE-SHOT endpoint for bots
     │
     ├─► NLP LAYER (parser.py)
     │       spaCy NER + sentence-transformers intent
     │       Human text → ScoreRequest (Obligation[])
     │
     ├─► ENGINE (scorer.py)
     │       Pure deterministic math. Zero LLM.
     │       Consequence Score formula (3 steps)
     │       Greedy constraint solver
     │       Days-to-Zero projection
     │       ScoreRequest → EngineResult
     │
     └─► NLP LAYER (narrate)
             EngineResult → human text
             Channels: whatsapp / web / voice
     │
     ▼
Twilio → WhatsApp reply
```

---

## Files

```
flowguard/
├── engine/
│   ├── models.py      ← All Pydantic schemas (shared contract)
│   └── scorer.py      ← Deterministic CS formula + solver
├── nlp/
│   └── parser.py      ← spaCy parser + narrator + email drafter
├── api/
│   ├── main.py        ← FastAPI app + all endpoints
│   └── whatsapp_webhook.py ← Twilio WhatsApp handler
└── tests/
    └── test_engine.py ← pytest suite (50+ tests)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download spaCy model
python -m spacy download en_core_web_sm

# 3. Run the server
uvicorn flowguard.api:app --host 0.0.0.0 --port 8000 --reload

# 4. Open docs
open http://localhost:8000/docs

# 5. Run tests
pytest flowguard/tests/ -v
```

---

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/health` | Check spaCy / sentence-transformers loaded |
| `POST` | `/parse` | Human text → ScoreRequest |
| `POST` | `/score` | ScoreRequest → EngineResult (pure engine) |
| `POST` | `/narrate` | EngineResult → human text |
| `POST` | `/pipeline` | **WhatsApp bot endpoint** (one-shot) |
| `POST` | `/email` | Draft negotiation email |
| `POST` | `/whatif` | Scenario analysis |
| `GET`  | `/audit/{run_id}` | Retrieve audit trail |

---

## NLP Stack

**Recommended**: spaCy + sentence-transformers

| Library | Role | Model |
|---------|------|-------|
| spaCy | NER (ORG, MONEY, DATE), tokenisation | `en_core_web_sm` (12 MB) |
| sentence-transformers | Intent classification | `all-MiniLM-L6-v2` (80 MB) |

**Why not GPT/Claude for parsing?**
Because the engine contract requires determinism. An LLM returning slightly different JSON each run breaks the audit trail. spaCy + regex gives identical parses for identical inputs.

**Multilingual**: Add `python -m spacy download xx_ent_wiki_sm` for Tamil/Hindi.

---

## Consequence Score Formula

```
Step 1: blended = clamp(0.25·P + 0.25·U + 0.25·L + 0.15·C + 0.10·R − 0.08·F, 0, 1)
Step 2: pre_floor_cs = type_ceiling(category) × blended × 100
Step 3: cs = max(pre_floor_cs, domain_floor(category))
        + statutory urgent pin: if statutory & due within 24h → cs ≥ 95
```

| Band | Range | Action |
|------|-------|--------|
| LOW | 0–20 | DEFER |
| MEDIUM | 21–39 | NEGOTIATE |
| HIGH | 40–69 | PAY THIS WEEK |
| CRITICAL | 70–100 | PAY IMMEDIATELY |

---

## WhatsApp Setup (Twilio)

```bash
export TWILIO_ACCOUNT_SID=ACxxxxxxxx
export TWILIO_AUTH_TOKEN=xxxxxxxx
export TWILIO_WHATSAPP_FROM="whatsapp:+14155238886"
```

In Twilio console → WhatsApp Sandbox:
- Webhook URL: `https://your-ngrok-url.ngrok.io/whatsapp`
- Method: POST

User commands:
- Any obligation text → full analysis
- `FULL` → detailed report
- `EMAIL 1` → draft delay email for #1
- `WHATIF 30000 tomorrow` → scenario analysis
- `RESET` → clear session
- `HELP` → instructions

---

## Edge Cases Handled

| Edge Case | Handling |
|-----------|----------|
| Due date in the past | Urgency = 1.0 (maximum) |
| `max_deferral_days = 0` | Flexibility forced to FIXED |
| Statutory due within 24h | CS pinned ≥ 95 |
| Zero available cash | ESCALATE (fixed) / NEGOTIATE |
| Cascade chain (A blocks B blocks C) | Graph traversal for contagion score |
| Large statutory + tight cash | Bank-freeze contagion = 0.85 |
| Identical CS scores | Tie-broken by amount DESC |
| Penalty rate not provided | Confidence penalised (−0.05) |
| CS > 100 | Clamped to 100 |
| Empty obligations list | 422 error with clear message |
