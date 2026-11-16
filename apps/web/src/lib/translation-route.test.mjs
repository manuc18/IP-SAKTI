import test from "node:test";
import assert from "node:assert/strict";

import { createTranslationHandler } from "./translation-route.mjs";

function translationRequest(body) {
  return new Request("http://localhost/api/translate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

test("returns provider output for a valid supported request", async () => {
  const handler = createTranslationHandler({
    provider: { translate: async () => "Translated [1]" },
  });
  const response = await handler(
    translationRequest({
      text: "Answer [1]",
      sourceLanguage: "en",
      targetLanguage: "hi",
      protectedTerms: [],
    }),
  );

  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), {
    translatedText: "Translated [1]",
    sourceLanguage: "en",
    targetLanguage: "hi",
  });
});

test("maps missing credentials or provider failure to the stable English fallback", async () => {
  const handler = createTranslationHandler({
    provider: {
      async translate() {
        throw new Error("private provider detail");
      },
    },
    logger: { warn() {} },
  });
  const response = await handler(
    translationRequest({
      text: "नमस्ते",
      sourceLanguage: "hi",
      targetLanguage: "en",
    }),
  );

  assert.equal(response.status, 503);
  assert.deepEqual(await response.json(), {
    error: "TRANSLATION_UNAVAILABLE",
    message: "Translation is unavailable right now. You can continue in English.",
  });
});

test("rejects unsupported language codes before invoking the provider", async () => {
  let called = false;
  const handler = createTranslationHandler({
    provider: {
      async translate() {
        called = true;
        return "bad";
      },
    },
  });
  const response = await handler(
    translationRequest({
      text: "bonjour",
      sourceLanguage: "fr",
      targetLanguage: "en",
    }),
  );

  assert.equal(response.status, 400);
  assert.equal(called, false);
  assert.deepEqual(await response.json(), {
    error: "INVALID_TRANSLATION_REQUEST",
    message: "Choose one of the supported demo languages.",
  });
});

test("rejects oversized translation text before invoking the provider", async () => {
  let called = false;
  const handler = createTranslationHandler({
    provider: {
      async translate() {
        called = true;
        return "bad";
      },
    },
  });
  const response = await handler(
    translationRequest({
      text: "a".repeat(10_001),
      sourceLanguage: "en",
      targetLanguage: "hi",
    }),
  );

  assert.equal(response.status, 400);
  assert.equal(called, false);
});
