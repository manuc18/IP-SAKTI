import assert from "node:assert/strict";
import test from "node:test";

import {
  INSUFFICIENT_MESSAGE,
  NOT_COVERED_MESSAGE,
  describeResult,
} from "./result-status.mjs";

test("a greeting renders its reply with no confidence, citations or sources", () => {
  const view = describeResult({
    status: "SOCIAL",
    answer: null,
    rendered: "Hello! I'm IP-SAKTI.",
    evidence: [],
  });
  assert.equal(view.kind, "social");
  assert.equal(view.text, "Hello! I'm IP-SAKTI.");
  assert.equal(view.confidence, null);
  assert.equal(view.sources, "none");
  assert.doesNotMatch(view.text, /insufficient/i);
});

test("a clarification shows the question, not 'No answer was published'", () => {
  const view = describeResult({
    status: "NEEDS_CLARIFICATION",
    answer: null,
    clarification: { question: "Which area of IP law?", options: ["patents"] },
  });
  assert.equal(view.kind, "clarification");
  assert.equal(view.text, "Which area of IP law?");
  assert.deepEqual(view.options, ["patents"]);
  assert.equal(view.confidence, null);
});

test("no relevant evidence is reported as not covered, without source cards", () => {
  const view = describeResult({
    status: "ABSTAINED",
    answer: {
      confidence: "NONE",
      insufficient_evidence_reasons: [
        { code: "no_relevant_evidence" },
        { code: "no_citable_authority" },
      ],
    },
  });
  assert.equal(view.kind, "not_covered");
  assert.equal(view.text, NOT_COVERED_MESSAGE);
  assert.match(view.text, /couldn't find this topic/);
  assert.match(view.text, /being expanded/);
  assert.equal(view.sources, "none");
  assert.equal(view.confidence, null);
});

test("relevant but insufficient evidence keeps that distinction", () => {
  const view = describeResult({
    status: "ABSTAINED",
    answer: { insufficient_evidence_reasons: [{ code: "ungrounded_answer" }] },
  });
  assert.equal(view.kind, "insufficient");
  assert.equal(view.text, INSUFFICIENT_MESSAGE);
  assert.doesNotMatch(view.text, /couldn.t find this topic/);
  // Lexical relevance is weak (an IPC question can match "punishment" in the
  // Drugs and Cosmetics Act), so this must not claim the passages were related.
  assert.doesNotMatch(view.text, /related provisions/i);
  assert.match(view.text, /being expanded/);
  assert.equal(view.sources, "retrieved");
});

test("a provider failure is not presented as an evidence judgement", () => {
  const view = describeResult({
    status: "UNAVAILABLE",
    service_failure: { message: "The language provider is rate-limited right now." },
    answer: { confidence: "NONE" },
  });
  assert.equal(view.kind, "unavailable");
  assert.match(view.text, /rate-limited/);
  assert.equal(view.confidence, null);
});

test("an answered result keeps its text, confidence and cited sources", () => {
  const view = describeResult({
    status: "ANSWERED",
    answer: { answer: "It must be signed [1].", confidence: "MODERATE" },
  });
  assert.equal(view.kind, "answered");
  assert.equal(view.confidence, "MODERATE");
  assert.equal(view.sources, "cited");
});
