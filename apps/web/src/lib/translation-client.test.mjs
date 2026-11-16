import test from "node:test";
import assert from "node:assert/strict";

import { collectProtectedTerms } from "./translation-client.mjs";

test("collects legal metadata but never explanatory evidence or disclaimer text", () => {
  const result = {
    disclaimer: "This is not legal advice.",
    answer: {
      answer: "Explanatory answer [1].",
      citations: [
        {
          citation_text: "Trade Marks Rules, 2017 — Rule 23",
          locator: "Rule 23",
          source_document: {
            official_source_url: "https://ipindia.gov.in/rules",
          },
        },
      ],
    },
    evidence: [
      {
        document_name: "Trade Marks Rules, 2017",
        citation: "Trade Marks Rules, 2017 — Rule 23",
        locator_text: "Rule 23",
        official_source_url: "https://ipindia.gov.in/rules",
        text: "Retrieved source text must not be submitted for translation.",
      },
    ],
  };

  assert.deepEqual(collectProtectedTerms(result), [
    "Trade Marks Rules, 2017 — Rule 23",
    "https://ipindia.gov.in/rules",
    "Trade Marks Rules, 2017",
    "Rule 23",
  ]);
});
