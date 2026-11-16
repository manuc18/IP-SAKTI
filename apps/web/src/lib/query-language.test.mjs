import assert from "node:assert/strict";
import test from "node:test";

import {
  detectScriptLanguage,
  queryTranslationFailed,
  resolveQueryLanguage,
} from "./query-language.mjs";

const HINDI = "मैं अपनी फर्म को ट्रेडमार्क कैसे कराऊँ?";
const MIXED = "मैं अपनी firm को trademark कैसे कराऊँ?";
const ENGLISH = "how do I trademark my firm?";

test("pure Hindi and mixed Hindi-English queries resolve to the same source language", () => {
  assert.equal(resolveQueryLanguage(HINDI, "en"), "hi");
  assert.equal(resolveQueryLanguage(MIXED, "en"), "hi");
  assert.equal(resolveQueryLanguage(HINDI, "hi"), "hi");
  assert.equal(resolveQueryLanguage(MIXED, "hi"), "hi");
  assert.equal(resolveQueryLanguage("मैं ट्रेडमार्क कैसे रजिस्टर करूँ?", "en"), "hi");
  assert.equal(resolveQueryLanguage("ट्रेडमार्क रजिस्ट्रेशन कैसे करें?", "en"), "hi");
});

test("an English query is never translated when the dropdown is English", () => {
  assert.equal(resolveQueryLanguage(ENGLISH, "en"), "en");
  assert.equal(resolveQueryLanguage("What does Section 3(p) exclude?", "en"), "en");
});

test("Latin-script input keeps the dropdown language, as before", () => {
  assert.equal(resolveQueryLanguage(ENGLISH, "hi"), "hi");
});

test("a Devanagari dropdown language is more specific than the script", () => {
  assert.equal(resolveQueryLanguage(HINDI, "mr"), "mr");
  assert.equal(resolveQueryLanguage(HINDI, "sa"), "sa");
  assert.equal(resolveQueryLanguage(HINDI, "ta"), "hi");
});

test("other Indic scripts are detected", () => {
  assert.equal(detectScriptLanguage("காப்புரிமை என்றால் என்ன?"), "ta");
  assert.equal(detectScriptLanguage("ট্রেডমার্ক কীভাবে নিবন্ধন করব?"), "bn");
  assert.equal(detectScriptLanguage(ENGLISH), null);
});

test("an untranslated Indic query is a failure, never a retrieval query", () => {
  assert.equal(queryTranslationFailed(MIXED, MIXED), true);
  assert.equal(queryTranslationFailed(HINDI, ""), true);
  assert.equal(queryTranslationFailed(MIXED, "How do I get a trademark for my firm?"), false);
  assert.equal(queryTranslationFailed(ENGLISH, ENGLISH), false);
});
