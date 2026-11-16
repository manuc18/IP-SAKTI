# Multilingual Voice Demo Design

## Purpose

Add a lightweight multilingual, voice-first presentation layer to the existing
IP-SAKTI Next.js interface without modifying the legal corpus, retrieval,
reranking, evidence policy, answer provider, citation validator, grounding gate,
confidence calculation, abstention behavior, or legal-answering contract.

The browser remains responsible for speech recognition and speech synthesis.
Google Cloud Translation runs only in a server-side Next.js route and is hidden
behind a small provider interface so a future Bhashini implementation can replace
it without changing the user interface or the RAG integration.

## Existing System Boundaries

The hackathon interface is `apps/web`, served locally at
`http://localhost:3001`. It calls `POST /api/ask` through the existing API path.
The API returns the pipeline's completed result, including answer text,
citations, evidence, confidence, cautions, disclaimer, timings, and result
status. That response remains the source of truth.

The multilingual layer will not write to or import internal modules under
`ai/`. It will not recreate any retrieval, citation, grounding, or legal-answer
decision. Translation happens before the existing query call and after a final
answer has already passed the existing pipeline gates.

## Architecture

### Browser

The existing Next.js page gains:

- A language selector for English, Hindi, Sanskrit, Marathi, Tamil, Telugu,
  Bengali, Gujarati, Kannada, and Punjabi.
- Start/stop microphone controls backed by `SpeechRecognition` or
  `webkitSpeechRecognition`.
- A live interim transcript and an editable final question field.
- A listen button backed by `window.speechSynthesis`.
- Clear status messages for loading, microphone permission denial, unsupported
  speech recognition, translation failure, and RAG failure.
- The required demo-layer label and the existing legal disclaimer.

The UI pass will simplify the existing presentation rather than replace its
design language: group language, microphone, transcript, editable question,
and submit actions into one obvious interaction card; reduce competing borders
and badges; improve spacing and mobile stacking; and retain the evidence-first
answer, caution, source, and disclaimer hierarchy. No decorative redesign may
obscure status, confidence, citation navigation, or the difference between an
answer, abstention, and provider failure.

The page keeps the unmodified RAG response in `result`. A separate
`displayAnswer` value holds translated explanatory prose. Citation and source
components continue to render from `result`, ensuring translation cannot alter
their data.

### Translation API

`POST /api/translate` is a Next.js server route. Its request contains:

```json
{
  "text": "Text to translate",
  "sourceLanguage": "hi",
  "targetLanguage": "en",
  "protectedTerms": ["Trade Marks Rules, 2017", "Rule 23"]
}
```

Its success response contains:

```json
{
  "translatedText": "Translated text",
  "sourceLanguage": "hi",
  "targetLanguage": "en"
}
```

Failures use a stable, non-technical response:

```json
{
  "error": "TRANSLATION_UNAVAILABLE",
  "message": "Translation is unavailable right now. You can continue in English."
}
```

The route validates text length and supported language codes before calling the
provider. English-to-English requests are returned unchanged without invoking
Google.

### Provider Boundary

`TranslationProvider` exposes one operation:

```text
translate(text, sourceLanguage, targetLanguage, protectedTerms)
  -> translated text
```

`GoogleCloudTranslationProvider` implements the operation using the official
Google Cloud Translation Node.js client and Application Default Credentials.
No credential content is accepted in requests, imported by client code, logged,
or returned.

The provider divides input into protected and translatable segments. Protected
segments are never submitted to Google; translated segments are reassembled
around them in their original byte-for-byte form. Protected segments include:

- Citation markers such as `[1]`.
- Explicit protected terms supplied from returned citation and evidence fields,
  including statute/document names and locators.
- Section, article, rule, regulation, schedule, clause, and form identifiers.
- URLs, ISO-style and common written dates, and numeric ranges.

The answer translation request derives protected terms from the existing
response but sends only the answer text to the translation route. Evidence text,
source cards, cautions, confidence, status labels, jurisdiction, URLs, and the
disclaimer are never translated.

## Request Flow

1. The user selects a language and speaks or edits a question.
2. Browser speech recognition uses the selected language locale and writes
   interim/final transcript text into the editable input.
3. For English, the page calls the existing `/api/ask` directly.
4. For another language, the page calls `/api/translate` with English as the
   target, then sends only the returned English question to `/api/ask`.
5. The existing RAG service returns its normal result without multilingual
   fields or altered logic.
6. For an answered result, the page submits only the explanatory answer text to
   `/api/translate`, along with protected statute names and locators gathered
   from the existing citations/evidence.
7. The page renders the translated explanatory text while rendering every other
   field directly from the untouched RAG response.
8. The listen button speaks the displayed answer using the selected language
   locale.

## Failure Behavior

- Unsupported speech recognition disables microphone input but leaves typed
  input available.
- Permission denial stops recording and shows a direct instruction to allow
  microphone access or type the question.
- Query translation failure does not send a non-English question into the
  English-only retriever. It returns the UI to an English-ready state with a
  message asking the user to continue in English.
- Answer translation failure displays the original English answer and the
  non-technical translation message. The RAG result, citations, and disclaimer
  remain visible.
- RAG transport or service failures retain the existing `UNAVAILABLE`, safe
  abstention, and error distinctions. Translation does not turn a failure into
  an answer and does not retry a provider request.
- Missing or invalid Google credentials are treated as translation
  unavailability, not as an application crash.

## Credentials and Configuration

Local credentials are supplied only through:

```text
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
```

`.env.example` contains a placeholder path only. `.gitignore` covers the
specific local filename pattern used for Google service accounts plus common
service-account/credential JSON names. The provided credential file remains
outside the repository and is never read by the implementation workflow.

The existing Groq variables remain unchanged:

```text
GROQ_API_KEY=<secret>
GROQ_MODEL=openai/gpt-oss-120b
GROQ_TIMEOUT_SECONDS=10
```

Only placeholder secret values belong in tracked documentation.

## Deployment Compatibility

The translation route is part of the existing Next.js deployment, while the
existing catch-all route continues to proxy RAG calls to the Python deployment.
This avoids adding translation logic or credentials to browser bundles and does
not change the Python production image or RAG service contract.

For local development, `npm run dev` serves the Next.js interface at
`http://localhost:3001` and the existing gateway at `http://localhost:5001`.
The Python demo service remains at `http://127.0.0.1:8000`.

## Testing

Focused Node tests will cover:

- Supported-language validation and English passthrough.
- Protected-segment extraction and byte-for-byte reconstruction.
- Successful translation through an injected fake external client.
- Missing credentials/client failures mapped to the stable fallback response.
- The route never requiring live Google API calls in tests.

Existing Python tests will be run to confirm no regression in the legal-answer
pipeline. The Next.js production build and repository lint command will be run,
with any pre-existing toolchain issue reported rather than hidden.

Manual verification will cover typed and spoken input, each failure state,
translated display text, preserved citation navigation/source metadata, the
unchanged disclaimer, and speech synthesis.

## Explicit Non-Goals

- No changes to the legal corpus, frozen benchmarks, or provenance artifacts.
- No translation of retrieved source text or legal metadata.
- No speech vendor, speech backend, or paid speech API.
- No Bhashini implementation before API approval.
- No retry loop, fallback RAG provider, legal correctness claim, or legal advice.
- No large frontend or backend refactor.
