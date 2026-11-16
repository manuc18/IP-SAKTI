import assert from "node:assert/strict";
import test from "node:test";

import { DEMO_QUESTIONS, DEMO_QUESTION_TEXTS } from "./demo-questions.mjs";

test("the curated set is small, unique and fully documented", () => {
  assert.ok(DEMO_QUESTIONS.length >= 6 && DEMO_QUESTIONS.length <= 10);
  assert.equal(new Set(DEMO_QUESTION_TEXTS).size, DEMO_QUESTION_TEXTS.length);
  for (const entry of DEMO_QUESTIONS) {
    for (const field of ["question", "answerType", "source", "locator", "expectedConfidence", "demoSafe"]) {
      assert.ok(entry[field], `${entry.question} is missing ${field}`);
    }
    assert.ok(["HIGH", "MODERATE"].includes(entry.expectedConfidence));
    // Questions only: an answer must never be hardcoded here.
    assert.equal("answer" in entry, false);
  }
});

test("the landing chips and tutorial cover different IP regimes", () => {
  const firstFour = DEMO_QUESTIONS.slice(0, 4).map((entry) => entry.source);
  assert.equal(new Set(firstFour).size, 4);
});

test("the stale-source demo question is not in the curated set", () => {
  assert.ok(!DEMO_QUESTION_TEXTS.some((q) => /literary work/.test(q)));
});
