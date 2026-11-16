# Multilingual Voice Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a clean, voice-first multilingual layer to the existing Next.js demo while keeping the citation-grounded RAG result and all legal metadata authoritative and unchanged.

**Architecture:** A server-only Next.js route calls Google Cloud Translation through an injected `TranslationProvider`. The browser translates a non-English question before the existing `/api/ask` call and stores translated explanatory prose separately from the untouched RAG response; browser Web Speech APIs provide speech input and output.

**Tech Stack:** Next.js 14, React 18, Node.js built-in test runner, `@google-cloud/translate`, browser Web Speech API, Tailwind CSS.

**Spec:** `docs/superpowers/specs/2026-09-16-multilingual-voice-demo-design.md`

## Global Constraints

- Do not modify `ai/`, the legal corpus, benchmark artifacts, evidence policy, citation validator, grounding gate, confidence logic, or answer generation.
- `POST /api/ask` and its response remain unchanged and authoritative.
- Google credentials are server-side Application Default Credentials referenced only through `GOOGLE_APPLICATION_CREDENTIALS`.
- Never read, print, commit, upload, or hardcode service-account JSON contents.
- Translate only query text and final explanatory answer prose; never translate source/evidence text, citations, statute names, legal identifiers, dates, links, confidence, jurisdiction, cautions, or disclaimer.
- Browser speech recognition and `speechSynthesis` are the only speech services.
- Missing translation configuration must degrade to a clear English fallback without crashing or sending non-English text to the English-only RAG.
- Preserve the distinct `ANSWERED`, `ABSTAINED`, `UNAVAILABLE`, invalid-request, and transport-failure states.
- Keep the existing information-only/not-legal-advice disclaimer visible.
- Keep the UI pass minimal: simplify hierarchy and spacing without obscuring evidence or safety states.

---

### Task 1: Translation provider and protected-segment behavior

**Files:**
- Create: `apps/web/src/lib/translation.mjs`
- Create: `apps/web/src/lib/translation.test.mjs`
- Modify: `apps/web/package.json`
- Modify: `package-lock.json`

**Interfaces:**
- Produces: `SUPPORTED_LANGUAGE_CODES: ReadonlySet<string>`
- Produces: `splitProtectedSegments(text: string, protectedTerms?: string[]): Array<{ text: string, protected: boolean }>`
- Produces: `GoogleCloudTranslationProvider.translate({ text, sourceLanguage, targetLanguage, protectedTerms }): Promise<string>`
- Consumes: an injected client supporting `translate(string[], { from: string, to: string, format: "text" })`.

- [ ] **Step 1: Add the test command and official dependency**

Update the web package scripts and dependencies:

```json
{
  "scripts": {
    "test": "node --test src/lib/*.test.mjs"
  },
  "dependencies": {
    "@google-cloud/translate": "^8.5.1"
  }
}
```

Run `npm install` from the repository root to update `package-lock.json`.

- [ ] **Step 2: Write failing protection and passthrough tests**

Create tests using `node:test` and `node:assert/strict`. The production mutations
these tests must catch are: sending protected legal metadata to Google, losing a
citation during reconstruction, and calling Google for English passthrough.

```js
import test from "node:test";
import assert from "node:assert/strict";
import {
  GoogleCloudTranslationProvider,
  splitProtectedSegments,
} from "./translation.mjs";

test("keeps citations, statute names, locators, dates, and URLs out of translated segments", () => {
  const text = "Under Trade Marks Rules, 2017, Rule 23 applies on 2026-09-16 [1] https://example.test/source.";
  const segments = splitProtectedSegments(text, ["Trade Marks Rules, 2017", "Rule 23"]);
  const protectedText = segments.filter((part) => part.protected).map((part) => part.text);
  assert.deepEqual(protectedText, [
    "Trade Marks Rules, 2017",
    "Rule 23",
    "2026-09-16",
    "[1]",
    "https://example.test/source",
  ]);
});

test("translates only explanatory segments and reconstructs protected text byte for byte", async () => {
  const calls = [];
  const client = {
    async translate(parts, options) {
      calls.push({ parts, options });
      return [parts.map((part) => `हिन्दी:${part}`)];
    },
  };
  const provider = new GoogleCloudTranslationProvider({ client });
  const result = await provider.translate({
    text: "The application is signed under Trade Marks Rules, 2017 — Rule 23 [1].",
    sourceLanguage: "en",
    targetLanguage: "hi",
    protectedTerms: ["Trade Marks Rules, 2017", "Rule 23"],
  });

  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].options, { from: "en", to: "hi", format: "text" });
  assert.ok(!calls[0].parts.join("").includes("Trade Marks Rules, 2017"));
  assert.equal(result.includes("Trade Marks Rules, 2017 — Rule 23 [1]"), true);
});

test("returns English text unchanged without calling the external client", async () => {
  const client = { translate: async () => { throw new Error("must not be called"); } };
  const provider = new GoogleCloudTranslationProvider({ client });
  assert.equal(await provider.translate({
    text: "Already English [1].",
    sourceLanguage: "en",
    targetLanguage: "en",
  }), "Already English [1].");
});
```

- [ ] **Step 3: Run the focused test and verify RED**

Run: `npm test --workspace web`

Expected: FAIL because `translation.mjs` does not exist.

- [ ] **Step 4: Implement the minimal provider**

Implement:

```js
import { v2 as translateV2 } from "@google-cloud/translate";

export const SUPPORTED_LANGUAGE_CODES = new Set([
  "en", "hi", "sa", "mr", "ta", "te", "bn", "gu", "kn", "pa",
]);

const BUILT_IN_PROTECTED = [
  /\[\d+\]/g,
  /https?:\/\/[^\s]+/g,
  /\b\d{4}-\d{2}-\d{2}\b/g,
  /\b(?:Section|Article|Rule|Regulation|Schedule|Clause|Form)\s+[A-Za-z0-9()./-]+/gi,
];

export function splitProtectedSegments(text, protectedTerms = []) {
  // Collect exact non-overlapping ranges from escaped supplied terms and the
  // built-in expressions, prefer the longest range at the same offset, and
  // return the full input as alternating protected/unprotected segments.
}

export class GoogleCloudTranslationProvider {
  constructor({ client } = {}) {
    this.client = client || new translateV2.Translate();
  }

  async translate({ text, sourceLanguage, targetLanguage, protectedTerms = [] }) {
    if (sourceLanguage === targetLanguage) return text;
    const segments = splitProtectedSegments(text, protectedTerms);
    const indexes = segments.flatMap((segment, index) =>
      !segment.protected && /\p{L}/u.test(segment.text) ? [index] : []);
    if (!indexes.length) return text;
    const values = indexes.map((index) => segments[index].text);
    const [translated] = await this.client.translate(values, {
      from: sourceLanguage,
      to: targetLanguage,
      format: "text",
    });
    const translatedValues = Array.isArray(translated) ? translated : [translated];
    indexes.forEach((segmentIndex, translatedIndex) => {
      segments[segmentIndex] = { text: translatedValues[translatedIndex], protected: false };
    });
    return segments.map((segment) => segment.text).join("");
  }
}
```

Fill the range collector explicitly with `RegExp.exec`, `match.index`, escaped
exact terms sorted longest-first, overlap rejection, and punctuation-safe URL
handling so the literal expected values pass.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run: `npm test --workspace web`

Expected: 3 tests pass, 0 fail.

- [ ] **Step 6: Commit the provider**

```bash
git add apps/web/package.json package-lock.json apps/web/src/lib/translation.mjs apps/web/src/lib/translation.test.mjs
git commit -m "feat: add protected Google translation adapter"
```

---

### Task 2: Server-side translation endpoint and fallback contract

**Files:**
- Create: `apps/web/src/lib/translation-route.mjs`
- Create: `apps/web/src/lib/translation-route.test.mjs`
- Create: `apps/web/src/app/api/translate/route.js`

**Interfaces:**
- Consumes: a provider with `translate(request): Promise<string>`.
- Produces: `createTranslationHandler({ provider }): (request: Request) => Promise<Response>`.
- Produces: `POST /api/translate` with the request and response contract from the design specification.

- [ ] **Step 1: Write failing route behavior tests**

The production mutations these tests must catch are: unsupported languages
reaching Google, leaking provider errors to the UI, accepting oversized input,
and changing fallback copy.

```js
import test from "node:test";
import assert from "node:assert/strict";
import { createTranslationHandler } from "./translation-route.mjs";

const request = (body) => new Request("http://localhost/api/translate", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

test("returns provider output for a valid supported request", async () => {
  const handler = createTranslationHandler({
    provider: { translate: async () => "Translated [1]" },
  });
  const response = await handler(request({
    text: "Answer [1]", sourceLanguage: "en", targetLanguage: "hi",
    protectedTerms: [],
  }));
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), {
    translatedText: "Translated [1]", sourceLanguage: "en", targetLanguage: "hi",
  });
});

test("maps missing credentials or provider failure to the stable English fallback", async () => {
  const handler = createTranslationHandler({
    provider: { translate: async () => { throw new Error("private provider detail"); } },
  });
  const response = await handler(request({
    text: "नमस्ते", sourceLanguage: "hi", targetLanguage: "en",
  }));
  assert.equal(response.status, 503);
  assert.deepEqual(await response.json(), {
    error: "TRANSLATION_UNAVAILABLE",
    message: "Translation is unavailable right now. You can continue in English.",
  });
});

test("rejects unsupported language codes before invoking the provider", async () => {
  let called = false;
  const handler = createTranslationHandler({
    provider: { translate: async () => { called = true; return "bad"; } },
  });
  const response = await handler(request({
    text: "bonjour", sourceLanguage: "fr", targetLanguage: "en",
  }));
  assert.equal(response.status, 400);
  assert.equal(called, false);
});
```

- [ ] **Step 2: Run route tests and verify RED**

Run: `npm test --workspace web`

Expected: provider tests pass and route tests fail because
`translation-route.mjs` does not exist.

- [ ] **Step 3: Implement validation and fallback mapping**

Implement `createTranslationHandler` to:

```text
parse JSON
require non-empty string text of at most 10,000 characters
require sourceLanguage and targetLanguage in SUPPORTED_LANGUAGE_CODES
accept protectedTerms only as an array of non-empty strings, capped at 100 entries
call provider.translate with only normalized fields
return translatedText on success
return HTTP 400 with INVALID_TRANSLATION_REQUEST for invalid input
return HTTP 503 with TRANSLATION_UNAVAILABLE and stable public copy on any provider error
never include exception text in the response
```

Create `route.js` as a thin composition root:

```js
import { GoogleCloudTranslationProvider } from "../../../lib/translation.mjs";
import { createTranslationHandler } from "../../../lib/translation-route.mjs";

const handler = createTranslationHandler({
  provider: new GoogleCloudTranslationProvider(),
});

export const runtime = "nodejs";
export const POST = handler;
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `npm test --workspace web`

Expected: all translation provider and route tests pass.

- [ ] **Step 5: Commit the endpoint**

```bash
git add apps/web/src/lib/translation-route.mjs apps/web/src/lib/translation-route.test.mjs apps/web/src/app/api/translate/route.js
git commit -m "feat: expose server-side translation endpoint"
```

---

### Task 3: Language configuration and browser voice controls

**Files:**
- Create: `apps/web/src/lib/languages.mjs`
- Modify: `apps/web/src/app/page.jsx`

**Interfaces:**
- Produces: `LANGUAGES` entries with `{ code, label, nativeLabel, speechLocale }`.
- Consumes: `SpeechRecognition`/`webkitSpeechRecognition` and `speechSynthesis` only in the browser.
- Produces: selected language, listening state, interim transcript, permission/unsupported messages, and speak/stop behavior.

- [ ] **Step 1: Add the ten-language configuration**

Create literal entries:

```js
export const LANGUAGES = [
  { code: "en", label: "English", nativeLabel: "English", speechLocale: "en-IN" },
  { code: "hi", label: "Hindi", nativeLabel: "हिन्दी", speechLocale: "hi-IN" },
  { code: "sa", label: "Sanskrit", nativeLabel: "संस्कृतम्", speechLocale: "sa-IN" },
  { code: "mr", label: "Marathi", nativeLabel: "मराठी", speechLocale: "mr-IN" },
  { code: "ta", label: "Tamil", nativeLabel: "தமிழ்", speechLocale: "ta-IN" },
  { code: "te", label: "Telugu", nativeLabel: "తెలుగు", speechLocale: "te-IN" },
  { code: "bn", label: "Bengali", nativeLabel: "বাংলা", speechLocale: "bn-IN" },
  { code: "gu", label: "Gujarati", nativeLabel: "ગુજરાતી", speechLocale: "gu-IN" },
  { code: "kn", label: "Kannada", nativeLabel: "ಕನ್ನಡ", speechLocale: "kn-IN" },
  { code: "pa", label: "Punjabi", nativeLabel: "ਪੰਜਾਬੀ", speechLocale: "pa-IN" },
];
```

- [ ] **Step 2: Add browser-only recognition lifecycle**

In `page.jsx`, add refs for recognition and speech utterance, feature detection
inside `useEffect`, and cleanup that stops recognition, cancels synthesis, and
clears existing timers. Configure:

```js
recognition.lang = selectedLanguage.speechLocale;
recognition.continuous = true;
recognition.interimResults = true;
```

On each result, join final chunks into the editable `question` and show interim
chunks in the live transcript area. Map `not-allowed` and `service-not-allowed`
to the microphone-permission message; map missing constructors to the
unsupported-browser message. Start and stop only on explicit button clicks.

- [ ] **Step 3: Add speech synthesis behavior**

Create an utterance from `displayAnswer || answerObj.answer`, set its `lang` to
the selected locale, cancel any current utterance before speaking, and expose a
button that toggles between listen and stop. Do not send speech text to any API.

- [ ] **Step 4: Run the production build**

Run: `npm run build --workspace web`

Expected: Next.js build exits 0 without server/browser global reference errors.

- [ ] **Step 5: Commit voice controls**

```bash
git add apps/web/src/lib/languages.mjs apps/web/src/app/page.jsx
git commit -m "feat: add multilingual browser voice controls"
```

---

### Task 4: Translate around the existing RAG call without mutating its result

**Files:**
- Modify: `apps/web/src/app/page.jsx`

**Interfaces:**
- Consumes: `POST /api/translate` and the existing `${API_BASE_URL}/api/ask`.
- Produces: `displayAnswer: string`, `translationNotice: string | null`, and an unchanged `result` object.

- [ ] **Step 1: Add a translation fetch helper and protected-term collector**

Add `translateText({ text, sourceLanguage, targetLanguage, protectedTerms })`
that calls relative `/api/translate`, checks `response.ok`, and throws an error
carrying only the returned public message.

Add `collectProtectedTerms(result)` returning unique, non-empty strings from:

```text
result.answer.citations[].citation_text
result.answer.citations[].locator
result.answer.citations[].source_document.official_source_url
result.evidence[].document_name
result.evidence[].citation
result.evidence[].locator_text
result.evidence[].official_source_url
```

Do not include `result.disclaimer`, evidence text, cautions, confidence, or
jurisdiction in the translation request at all.

- [ ] **Step 2: Integrate pre-query translation**

Before the existing ask fetch:

```js
const englishQuestion = selectedLanguage.code === "en"
  ? q
  : await translateText({
      text: q,
      sourceLanguage: selectedLanguage.code,
      targetLanguage: "en",
      protectedTerms: [],
    });
```

Send `englishQuestion` in the unchanged `{ question }` body. If this translation
fails, stop before `/api/ask`, set the stable translation notice, and leave the
typed question editable. Do not label the incident as a RAG failure.

- [ ] **Step 3: Integrate post-answer translation**

After storing the untouched response with `setResult(data)`, translate only
`data.answer.answer` when the response is `ANSWERED` and the selected language
is not English. Store the output only in `displayAnswer`. On failure, set
`displayAnswer` to the original English answer and show the stable notice.

Abstentions, cautions, sources, confidence, jurisdiction, disclaimer, and
provider failures render from `data` exactly as returned and are not sent for
translation.

- [ ] **Step 4: Render translated prose and optional returned jurisdiction**

Pass `displayAnswer || answerObj.answer` to the existing citation-aware prose
renderer. Add a conditional jurisdiction badge only when the current response
actually contains `result.jurisdiction` or `result.answer.jurisdiction`; show
the exact value without deriving, translating, or defaulting it.

- [ ] **Step 5: Run tests and build**

Run:

```bash
npm test --workspace web
npm run build --workspace web
```

Expected: all focused tests pass and the Next.js build exits 0.

- [ ] **Step 6: Commit RAG integration**

```bash
git add apps/web/src/app/page.jsx
git commit -m "feat: translate around the grounded query flow"
```

---

### Task 5: Clean and simplify the hackathon interface

**Files:**
- Modify: `apps/web/src/app/page.jsx`
- Modify: `apps/web/src/app/globals.css`

**Interfaces:**
- Preserves: all existing evidence cards, citation links, cautions, answer states, confidence, timings, and disclaimer.
- Produces: one responsive interaction card for language, voice, transcript, question, and submit actions.

- [ ] **Step 1: Consolidate the input hierarchy**

Place the language dropdown at the top of the existing question card, followed
by one prominent microphone button and a compact live transcript region, then
the editable textarea and submit button. Add the exact label:

```text
Demo multilingual layer: Google Cloud Translation + browser voice. Bhashini integration will be enabled after API approval.
```

Keep sample questions secondary and do not auto-submit when a sample is chosen.

- [ ] **Step 2: Reduce visual noise**

Use one accent color for primary actions, reduce redundant borders/background
fills around metadata, keep status colors semantic, shorten button labels, and
retain square/subtly rounded geometry. Keep body width readable and ensure
controls stack to full width below the `sm` breakpoint.

- [ ] **Step 3: Add accessible interaction states**

Add explicit labels, `aria-live="polite"` to transcript/status text,
`aria-pressed` to microphone and listen toggles, visible focus styles, disabled
states during translation/RAG loading, and text labels alongside icons.

- [ ] **Step 4: Verify responsive production output**

Run: `npm run build --workspace web`

Then inspect `http://localhost:3001` at desktop and narrow mobile widths. Verify
no horizontal overflow and that source text remains readable.

- [ ] **Step 5: Commit the UI pass**

```bash
git add apps/web/src/app/page.jsx apps/web/src/app/globals.css
git commit -m "style: simplify multilingual demo interface"
```

---

### Task 6: Credential safety and operator documentation

**Files:**
- Modify: `.env.example`
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `DEMO.md`

**Interfaces:**
- Documents: `GOOGLE_APPLICATION_CREDENTIALS`, existing Groq variables, exact local commands, and manual test flow.

- [ ] **Step 1: Add safe environment configuration**

Add only this placeholder to `.env.example`:

```dotenv
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
```

Keep the existing `GROQ_API_KEY=gsk_replace_me` placeholder. Do not copy the
user's real key or credential path into tracked files.

- [ ] **Step 2: Extend credential ignore rules**

Add targeted patterns:

```gitignore
# Cloud service-account credentials
*-service-account*.json
*service_account*.json
*serviceaccount*.json
google-credentials*.json
gcp-credentials*.json
geometric-petal-*.json
```

- [ ] **Step 3: Document local setup and demo checks**

Add concise README and DEMO sections covering:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
npm install
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
.venv/bin/python -m app.server --provider extractive --no-dense
npm run dev
```

Document `http://localhost:3001` as the multilingual UI,
`http://127.0.0.1:8000` as the Python demo/API, Chrome/Edge as the recommended
speech-recognition browsers, the English fallback behavior, and the manual
Hindi/English/citation/listen checks.

- [ ] **Step 4: Verify no credential is tracked or present in the diff**

Run:

```bash
git status --short
git diff --check
git diff -- .env.example .gitignore README.md DEMO.md
git ls-files '*.json' | rg -i 'service.?account|credentials|geometric-petal' || true
```

Expected: only placeholder configuration appears; no credential JSON is tracked.

- [ ] **Step 5: Commit configuration and docs**

```bash
git add .env.example .gitignore README.md DEMO.md
git commit -m "docs: add multilingual demo setup"
```

---

### Task 7: Full verification and local demo handoff

**Files:**
- Verify only; modify implementation files solely if a test exposes a defect.

**Interfaces:**
- Verifies: Node adapter/route behavior, Next build, repository lint behavior, Python legal pipeline regression suite, and local URLs.

- [ ] **Step 1: Run focused JavaScript tests**

Run: `npm test --workspace web`

Expected: all translation tests pass with no Google network call.

- [ ] **Step 2: Run frontend lint and build**

Run:

```bash
npm run lint
npm run build
```

Expected: both exit 0. If the repository's existing `next lint` command is
incompatible with its installed Next.js version, report the exact pre-existing
tooling failure and run the build plus targeted tests; do not conceal it.

- [ ] **Step 3: Run Python regression tests**

Run: `.venv/bin/python -m pytest -q`

Expected: the existing legal pipeline suite passes with no modifications under
`ai/`, `data/`, benchmark artifacts, or grounding/citation policy files.

- [ ] **Step 4: Start the local stack without exposing secrets**

In one terminal:

```bash
.venv/bin/python -m app.server --provider extractive --no-dense --host 127.0.0.1 --port 8000
```

In another terminal, after setting credentials in the shell or an ignored local
environment file:

```bash
npm run dev
```

Do not print environment values in logs or commands captured for the handoff.

- [ ] **Step 5: Verify routes and UI**

Check:

```text
GET http://127.0.0.1:8000/api/health -> 200
GET http://localhost:3001 -> 200
POST http://localhost:3001/api/translate with en->en -> 200 and unchanged text
```

Manually verify Hindi speech/type to English RAG to protected translated answer,
English direct query, citation click/source expansion, untranslated metadata and
disclaimer, listen/stop, denied microphone, and missing-credential fallback.

- [ ] **Step 6: Audit the final diff and branch**

Run:

```bash
git status --short --branch
git diff main...HEAD --stat
git diff main...HEAD -- ai data
git log --oneline --decorate main..HEAD
```

Expected: branch is `manu/multilingual-voice-demo`; no diff exists under `ai/`
or `data/`; only scoped multilingual UI, server translation, tests,
configuration, documentation, and plan/spec files changed.
