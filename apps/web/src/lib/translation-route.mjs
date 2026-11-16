import { SUPPORTED_LANGUAGE_CODES } from "./translation.mjs";

const MAX_TEXT_LENGTH = 10_000;
const MAX_PROTECTED_TERMS = 100;
const MAX_PROTECTED_TERM_LENGTH = 500;

const FALLBACK_BODY = Object.freeze({
  error: "TRANSLATION_UNAVAILABLE",
  message: "Translation is unavailable right now. You can continue in English.",
});

function json(body, status) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

function invalid(message) {
  return json(
    { error: "INVALID_TRANSLATION_REQUEST", message },
    400,
  );
}

function validateBody(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return { error: "Translation request must be a JSON object." };
  }

  const { text, sourceLanguage, targetLanguage } = body;
  if (typeof text !== "string" || text.trim().length === 0) {
    return { error: "Enter text to translate." };
  }
  if (text.length > MAX_TEXT_LENGTH) {
    return { error: "Text is too long to translate." };
  }
  if (
    !SUPPORTED_LANGUAGE_CODES.has(sourceLanguage) ||
    !SUPPORTED_LANGUAGE_CODES.has(targetLanguage)
  ) {
    return { error: "Choose one of the supported demo languages." };
  }

  const protectedTerms = body.protectedTerms ?? [];
  if (
    !Array.isArray(protectedTerms) ||
    protectedTerms.length > MAX_PROTECTED_TERMS ||
    protectedTerms.some(
      (term) =>
        typeof term !== "string" ||
        term.trim().length === 0 ||
        term.length > MAX_PROTECTED_TERM_LENGTH,
    )
  ) {
    return { error: "Protected legal terms are not valid." };
  }

  return {
    value: {
      text,
      sourceLanguage,
      targetLanguage,
      protectedTerms: [...new Set(protectedTerms)],
    },
  };
}

export function createTranslationHandler({ provider, logger = console }) {
  if (!provider || typeof provider.translate !== "function") {
    throw new TypeError("A translation provider is required.");
  }

  return async function handleTranslation(request) {
    let body;
    try {
      body = await request.json();
    } catch {
      return invalid("Translation request must be valid JSON.");
    }

    const validation = validateBody(body);
    if (validation.error) return invalid(validation.error);

    try {
      const translatedText = await provider.translate(validation.value);
      return json(
        {
          translatedText,
          sourceLanguage: validation.value.sourceLanguage,
          targetLanguage: validation.value.targetLanguage,
        },
        200,
      );
    } catch (err) {
      logger.warn(`[translation] Google Cloud Translation is unavailable: ${err?.message || err}`);
      return json(FALLBACK_BODY, 503);
    }
  };
}
