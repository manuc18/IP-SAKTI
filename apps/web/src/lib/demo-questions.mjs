// The one curated list of demo questions. The tutorial (first 3), the landing
// chips (first 4) and the rotating input placeholder (all) read only this.
//
// Only the questions are hardcoded. Every answer still comes from the live
// pipeline, so a click behaves exactly like a typed query. Each entry records
// what it produced when run against the current pipeline (Groq provider, answer
// cache off, 2026-09-29), so a regression is visible against this file.
//
// Every cited source below is CURRENT and none raised a conflict. `note` marks
// the entries whose response also carries the mild "evidence mixes current and
// superseded instruments … not flagged as a conflict" caution, caused by an
// uncited STALE passage in the retrieved set, not by the cited provision.
//
// Excluded after testing: "How must an application for registration of a trade
// mark be signed?" (answers, but its only cited passage, TM Rules r.23, is
// flagged EXTRACTION_DAMAGED) and "What is the term of copyright in a literary
// work…" (Copyright Act 1957 is STALE; it is the stale-caution demo, not a
// clean one).

export const DEMO_QUESTIONS = Object.freeze([
  {
    question: "What is the term of a patent in India?",
    answerType: "single-source factual",
    source: "Patents Act, 1970",
    locator: "Section 53",
    expectedConfidence: "MODERATE",
    demoSafe: "Short, unambiguous answer from one current provision; no caution.",
  },
  {
    question: "What is the duration of the registration of a trade mark and when must it be renewed?",
    answerType: "multi-source procedural",
    source: "Trade Marks Act, 1999 + Trade Marks Rules, 2017",
    locator: "Section 25; Rule 57",
    expectedConfidence: "MODERATE",
    demoSafe: "Shows [1] and [4] mapping to an Act and its Rules; both current.",
    note: "mixed-currency caution",
  },
  {
    question: "How is an application for registration of copyright made under the Copyright Rules, 2013?",
    answerType: "single-source procedural",
    source: "Copyright Rules, 2013",
    locator: "Rule 70",
    expectedConfidence: "HIGH",
    demoSafe: "Provenance level A, current; avoids the stale Copyright Act, 1957.",
    note: "mixed-currency caution",
  },
  {
    question: "Who may apply for the registration of a geographical indication?",
    answerType: "single-source factual",
    source: "Geographical Indications of Goods (Registration and Protection) Act, 1999",
    locator: "Section 11",
    expectedConfidence: "MODERATE",
    demoSafe: "Clear one-sentence answer; shows GI coverage.",
    note: "mixed-currency caution",
  },
  {
    question: "Is the mere discovery of a new form of a known substance patentable in India?",
    answerType: "single-source legal test",
    source: "Patents Act, 1970",
    locator: "Section 3(d)",
    expectedConfidence: "MODERATE",
    demoSafe: "Well-known provision; answer cites the rank-4 passage, not rank 1.",
  },
  {
    question: "What does section 3(p) of the Patents Act, 1970 exclude from patentability?",
    answerType: "single-source identifier lookup",
    source: "Patents Act, 1970",
    locator: "Section 3(p)",
    expectedConfidence: "MODERATE",
    demoSafe: "Provision-number query; traditional-knowledge exclusion; no caution.",
  },
  {
    question: "What designs cannot be registered under the Designs Act, 2000?",
    answerType: "single-source factual",
    source: "Designs Act, 2000",
    locator: "Section 4",
    expectedConfidence: "MODERATE",
    demoSafe: "Shows designs coverage; one current provision.",
    note: "mixed-currency caution",
  },
  {
    question: "Under which provision may a compulsory licence be granted over a registered plant variety?",
    answerType: "single-source provision lookup",
    source: "Protection of Plant Varieties and Farmers' Rights Rules, 2003",
    locator: "Rule 71",
    expectedConfidence: "HIGH",
    demoSafe: "Provenance level A, current; no caution.",
  },
]);

export const DEMO_QUESTION_TEXTS = Object.freeze(
  DEMO_QUESTIONS.map((entry) => entry.question),
);
