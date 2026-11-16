// How one backend result is presented. Pure, so every status is testable.
//
// The backend already distinguishes these states (`app/pipeline.py`):
//   SOCIAL               greeting/help; no retrieval, reply in `rendered`
//   NEEDS_CLARIFICATION  no regime named; nothing retrieved yet
//   UNAVAILABLE          provider failure (rate limit, network)
//   ABSTAINED            retrieval ran; nothing publishable
//   ANSWERED             grounded, validated, cited
// Only ANSWERED carries a confidence level, citations or source cards.

export const NOT_COVERED_MESSAGE =
  "I couldn't find this topic in the current IP-SAKTI corpus. " +
  "The system currently covers a limited set of statutory-law sources and the corpus is being expanded.";

export const INSUFFICIENT_MESSAGE =
  "I couldn't find sufficient authoritative evidence in the current IP-SAKTI corpus to answer this reliably. " +
  "The passages retrieved did not support a fully cited answer, so none is shown. " +
  "This system currently covers a limited set of statutory-law sources, and the corpus is being expanded; " +
  "this question may be outside the sources currently available.";

// Reason codes that mean "nothing in the corpus addresses this" rather than
// "relevant passages were found but did not support an answer".
const NOT_COVERED_CODES = new Set(["no_evidence", "no_relevant_evidence"]);

export function abstentionKind(answer) {
  const codes = (answer?.insufficient_evidence_reasons || []).map((r) =>
    String(r?.code || "").toLowerCase(),
  );
  return codes.some((code) => NOT_COVERED_CODES.has(code))
    ? "not_covered"
    : "insufficient";
}

export function describeResult(data) {
  const status = data?.status;
  const answer = data?.answer || {};

  if (status === "SOCIAL") {
    return { kind: "social", text: data.rendered || "", confidence: null, sources: "none" };
  }
  if (status === "NEEDS_CLARIFICATION") {
    const clarification = data.clarification || {};
    return {
      kind: "clarification",
      text: clarification.question || "",
      options: Array.isArray(clarification.options) ? clarification.options : [],
      confidence: null,
      sources: "none",
    };
  }
  if (status === "UNAVAILABLE") {
    return {
      kind: "unavailable",
      text:
        data.service_failure?.message ||
        "The answer service is unavailable right now. No answer was generated.",
      confidence: null,
      sources: "none",
    };
  }
  if (status === "ABSTAINED") {
    const kind = abstentionKind(answer);
    return {
      kind: kind === "not_covered" ? "not_covered" : "insufficient",
      text: kind === "not_covered" ? NOT_COVERED_MESSAGE : INSUFFICIENT_MESSAGE,
      confidence: null,
      // Retrieved-but-insufficient passages stay inspectable, labelled as not
      // cited. A not-covered result has nothing worth showing.
      sources: kind === "not_covered" ? "none" : "retrieved",
    };
  }
  return {
    kind: "answered",
    text: answer.answer || "",
    confidence: answer.confidence || null,
    sources: "cited",
  };
}
