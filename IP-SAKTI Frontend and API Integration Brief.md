# IP-SAKTI — DEV TEAM INTEGRATION BRIEF

## Objective

We are building the SIH prototype of **IP-SAKTI**, an evidence-first Intellectual Property legal assistant.

The core AI/RAG pipeline is being finalized separately.

Your responsibility is to build the **application layer and user experience around the pipeline**.

The architecture should be:

```text
USER
  │
  ▼
FRONTEND
  │
  │ POST JSON
  ▼
FLASK API
  │
  ▼
IP-SAKTI PIPELINE ENGINE
  │
  ├── Query Processing
  ├── Retrieval
  ├── Reranking
  ├── Evidence Selection
  ├── Grounded Answer Generation
  ├── Citation Validation
  └── Grounding Validation
  │
  ▼
NORMALIZED JSON RESPONSE
  │
  ▼
FRONTEND UI
```

---

# 1. TECH STACK

## Backend

Use:

```text
Python
Flask
Flask-CORS
```

Suggested structure:

```text
backend/
│
├── app.py
│
├── routes/
│   └── ask.py
│
├── services/
│   └── pipeline_service.py
│
├── schemas/
│   └── response_schema.py
│
└── requirements.txt
```

IMPORTANT:

The Flask server should be a **thin API layer**.

Do NOT duplicate:

- retrieval logic;
- reranking logic;
- answer generation logic;
- citation validation;
- grounding validation.

Those should remain inside the existing IP-SAKTI pipeline.

The Flask server should simply:

```text
Receive Question
      ↓
Validate Input
      ↓
Call Pipeline
      ↓
Receive Pipeline Result
      ↓
Normalize Result
      ↓
Return JSON
```

---

# 2. PRIMARY API

Create:

```text
POST /api/ask
```

Request:

```json
{
  "question": "How is an application for trademark registration filed?"
}
```

Optional future fields can include:

```json
{
  "question": "How is an application for trademark registration filed?",
  "session_id": "optional-session-id"
}
```

But for the SIH prototype, `question` is enough.

---

# 3. REQUEST VALIDATION

The API should reject:

```text
missing question
empty question
non-string question
extremely large input
```

Example failure:

```json
{
  "success": false,
  "status": "invalid_request",
  "message": "Please provide a valid legal question."
}
```

Do NOT send malformed input into the pipeline.

---

# 4. INTERNAL PIPELINE FLOW

The API must NOT contain the RAG logic itself.

Conceptually:

```python
result = pipeline.answer(question)
```

or whatever function/interface the core team exposes.

The frontend/API team should receive a stable pipeline result and convert it into a frontend-safe response.

Conceptually:

```text
Question
   ↓
pipeline.run(question)
   ↓
Raw Internal Result
   ↓
Response Normalizer
   ↓
Frontend JSON
```

---

# 5. MOST IMPORTANT REQUIREMENT: DO NOT DUMP RAW INTERNAL JSON DIRECTLY

The pipeline may produce deeply nested objects.

Example:

```text
result
 ├── retrieval
 │    ├── candidates
 │    ├── scores
 │    └── chunks
 │
 ├── reranking
 │    ├── scores
 │    └── ordering
 │
 ├── answer
 │    ├── text
 │    ├── citations
 │    └── confidence
 │
 ├── validation
 │    ├── citation
 │    └── grounding
 │
 └── diagnostics
      ├── versions
      ├── timings
      └── metadata
```

DO NOT simply return this to the frontend.

Instead create a **response adapter/normalizer**.

The frontend should receive a stable, simple JSON contract.

---

# 6. STANDARD API RESPONSE CONTRACT

Successful answer:

```json
{
  "success": true,

  "status": "answered",

  "question": "How is an application for trademark registration filed?",

  "answer": {
    "text": "An application for the registration of a trademark must be made in Form TM-A and signed by the applicant or his agent [1].",

    "confidence": "MODERATE"
  },

  "citations": [
    {
      "id": 1,

      "marker": "[1]",

      "document": "Trade Marks Rules, 2017",

      "locator": "Rule 23",

      "page": 59,

      "source_status": "CURRENT",

      "provenance_level": "B",

      "evidence_text": "An application for the registration of a trademark shall be made in Form TM-A and signed by the applicant or his agent."
    }
  ],

  "cautions": [],

  "meta": {
    "answer_status": "PUBLISHED"
  }
}
```

This is the contract the frontend should consume.

---

# 7. SAFE ABSTENTION RESPONSE

If the system cannot produce a sufficiently supported answer:

```json
{
  "success": true,

  "status": "abstained",

  "question": "User question",

  "answer": {
    "text": "I couldn't produce a sufficiently supported answer from the available legal evidence.",

    "confidence": null
  },

  "citations": [],

  "cautions": [
    "The available evidence was insufficient to publish a grounded answer."
  ],

  "meta": {
    "answer_status": "ABSTAINED"
  }
}
```

The frontend should clearly distinguish:

```text
ANSWERED
```

from:

```text
ABSTAINED
```

An abstention is not a system crash.

It is a valid safety behavior.

---

# 8. ERROR RESPONSE

If something genuinely fails:

```json
{
  "success": false,

  "status": "error",

  "message": "The IP-SAKTI service could not process the request.",

  "error_code": "PIPELINE_ERROR"
}
```

Do not expose:

- API keys;
- internal stack traces;
- sensitive paths;
- provider credentials.

Log internal errors on the backend only.

---

# 9. HOW TO HANDLE NESTED PIPELINE OUTPUT

The pipeline team may return deeply nested Python objects or dictionaries.

The frontend team should NOT depend directly on internal pipeline structure.

Create:

```text
normalize_response(raw_result)
```

Example:

```python
def normalize_response(raw_result):

    return {
        "success": True,
        "status": determine_status(raw_result),

        "question": raw_result.question,

        "answer": {
            "text": extract_answer_text(raw_result),
            "confidence": extract_confidence(raw_result)
        },

        "citations": extract_citations(raw_result),

        "cautions": extract_cautions(raw_result),

        "meta": {
            "answer_status": extract_answer_status(raw_result)
        }
    }
```

The exact field extraction depends on the final pipeline object.

The important rule is:

> **Internal pipeline format may change. The API response contract should remain stable.**

---

# 10. CITATION DATA

Each citation returned to the frontend should be normalized.

Use:

```json
{
  "id": 1,
  "marker": "[1]",
  "document": "Document Name",
  "locator": "Section / Rule",
  "page": 10,
  "source_status": "CURRENT",
  "provenance_level": "A",
  "evidence_text": "Relevant evidence passage..."
}
```

Frontend should render:

```text
[1] Trade Marks Rules, 2017
    Rule 23
    CURRENT
    View Evidence
```

When the user clicks:

```text
View Evidence
```

show the `evidence_text`.

Do not expose all internal retrieval metadata by default.

---

# 11. FRONTEND RESPONSE HANDLING

The frontend should handle three major states.

## STATE 1 — LOADING

Immediately after the user submits:

```text
IP-SAKTI is analyzing the legal evidence...
```

Show:

- loader;
- disabled submit button;
- optional progress animation.

Do NOT pretend the model is typing unless implemented intentionally.

---

## STATE 2 — ANSWERED

Render:

```text
ANSWER

<Answer text>

Confidence: Moderate

Sources
────────────────────

[1] Trade Marks Rules, 2017
Rule 23

[View Evidence]
```

Then allow evidence expansion.

---

## STATE 3 — ABSTAINED

Render clearly:

```text
I couldn't produce a sufficiently supported answer from the available legal evidence.
```

Then show:

```text
Why?

The system did not have enough validated evidence to publish a grounded answer.
```

This is an important differentiator of the product.

---

# 12. CAUTIONS

The API may return cautions.

Examples:

```text
The cited source is marked stale.
```

or:

```text
Some source evidence has weaker provenance.
```

Render cautions visibly but do NOT make them look like application errors.

Recommended:

```text
⚠ Source Notice

The cited source is marked stale.
Verify against the current official text.
```

---

# 13. FRONTEND TECH STACK

Use the team's fastest existing stack.

Recommended if starting fresh:

```text
React
Vite
Tailwind CSS
Axios or Fetch API
```

If Next.js is already available and working, use:

```text
Next.js
React
Tailwind CSS
```

DO NOT spend time migrating frameworks.

The priority is:

```text
WORKING DEMO
>
VISUAL POLISH
>
ARCHITECTURAL PERFECTION
```

---

# 14. FRONTEND COMPONENT STRUCTURE

Suggested:

```text
src/
│
├── components/
│   ├── QuestionInput.jsx
│   ├── AnswerCard.jsx
│   ├── CitationCard.jsx
│   ├── EvidenceModal.jsx
│   ├── CautionBanner.jsx
│   └── LoadingState.jsx
│
├── services/
│   └── api.js
│
├── pages/
│   └── Home.jsx
│
└── App.jsx
```

Main flow:

```text
Home
 │
 ├── QuestionInput
 │
 ├── LoadingState
 │
 ├── AnswerCard
 │
 ├── CautionBanner
 │
 └── CitationCard(s)
```

---

# 15. FRONTEND API CALL

Example:

```javascript
const response = await fetch(
  "http://localhost:5000/api/ask",
  {
    method: "POST",

    headers: {
      "Content-Type": "application/json"
    },

    body: JSON.stringify({
      question: userQuestion
    })
  }
);

const data = await response.json();
```

Then branch based on:

```javascript
data.success
data.status
```

Primary statuses:

```text
answered
abstained
invalid_request
error
```

Do NOT infer status from the presence of answer text.

Use the explicit API status.

---

# 16. CORS

The Flask API should allow the frontend origin.

For local development:

```text
Frontend:
http://localhost:5173

Backend:
http://localhost:5000
```

Use Flask-CORS.

Keep configuration simple for the SIH prototype.

---

# 17. BACKEND HEALTH ENDPOINT

Create:

```text
GET /health
```

Response:

```json
{
  "status": "ok",
  "service": "IP-SAKTI API"
}
```

Optional pipeline status:

```json
{
  "status": "ok",

  "service": "IP-SAKTI API",

  "pipeline": "ready"
}
```

This helps quickly demonstrate that the backend is alive.

---

# 18. OPTIONAL DEBUG ENDPOINT — DEVELOPMENT ONLY

If useful:

```text
POST /api/debug
```

This may return internal diagnostics such as:

- retrieved chunks;
- candidate ranks;
- reranking details;
- validation results;
- grounding diagnostics.

BUT:

This endpoint must be:

```text
development-only
```

Do NOT connect the main user UI directly to raw debug output.

For the SIH demo, internal pipeline diagnostics can be useful for explaining the architecture to judges.

---

# 19. IMPORTANT API DESIGN PRINCIPLE

Separate:

```text
INTERNAL PIPELINE RESULT
```

from:

```text
PUBLIC API RESPONSE
```

Architecture:

```text
PIPELINE

{
    deeply nested internal structures
}

        ↓

API ADAPTER

normalize_response()

        ↓

FRONTEND

{
    simple stable JSON
}
```

This prevents the frontend from breaking whenever internal pipeline structures change.

---

# 20. PERFORMANCE

Do NOT do any of the following during a normal API request unless required by the existing pipeline:

- re-ingest corpus;
- rebuild indexes;
- run benchmarks;
- run evaluation suites;
- download models;
- start background benchmark jobs.

Normal request:

```text
Question
↓
Existing Index
↓
Retrieve
↓
Rerank
↓
Answer
↓
Validate
↓
Return
```

---

# 21. TIMEOUT AND ERROR HANDLING

The frontend should have a reasonable request timeout.

If the backend takes too long:

```text
"The request is taking longer than expected. Please try again."
```

The backend should return meaningful HTTP codes.

Suggested:

```text
200 → answered or abstained

400 → invalid question

500 → internal processing failure

503 → pipeline/provider unavailable
```

Important:

An answer abstention should normally be:

```text
HTTP 200
```

because the system successfully processed the request.

---

# 22. DEMO REQUIREMENTS

Before final SIH demo, test:

### Test 1

Normal grounded answer.

Expected:

```text
ANSWERED
+
citations
+
evidence
```

### Test 2

Different IP category.

Expected:

```text
ANSWERED
+
different source
```

### Test 3

Safe abstention.

Expected:

```text
ABSTAINED
```

### Test 4

Source caution if available.

Expected:

```text
ANSWERED
+
STALE / provenance caution
```

---

# 23. TEAM RESPONSIBILITY SPLIT

## CORE PIPELINE TEAM

Responsible for:

```text
Retrieval
Reranking
Evidence
Answer Generation
Citation Validation
Grounding
Answer/Abstention Decision
```

## BACKEND/API TEAM

Responsible for:

```text
Flask Server
API Routes
Input Validation
Calling Pipeline
Response Normalization
Error Handling
Health Endpoint
CORS
```

## FRONTEND TEAM

Responsible for:

```text
Question Input
Loading State
API Integration
Answer Display
Confidence Display
Citation Display
Evidence Expansion
Caution Display
Error Handling
UI Polish
```

---

# 24. MOST IMPORTANT RULE

DO NOT MODIFY THE CORE PIPELINE CONTRACT TO MAKE FRONTEND INTEGRATION EASIER.

Instead:

```text
Core Pipeline
      ↓
Adapter
      ↓
Clean API JSON
      ↓
Frontend
```

The adapter is the boundary between the AI/legal pipeline and the application.

---

# 25. FINAL DELIVERABLE

The SIH prototype should allow:

```text
Judge opens IP-SAKTI
        ↓
Asks legal question
        ↓
System processes it
        ↓
Answer appears
        ↓
Citation is visible
        ↓
Judge opens evidence
        ↓
Judge sees provision/source
        ↓
System demonstrates evidence-first legal AI
```

The goal is not to build a production legal platform tonight.

The goal is to build a:

> **Fast, reliable, explainable and impressive SIH prototype demonstrating grounded Intellectual Property legal assistance.**