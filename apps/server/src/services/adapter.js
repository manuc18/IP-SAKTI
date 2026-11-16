/**
 * IP-SAKTI API Response Adapter
 *
 * Normalizes raw pipeline outcomes into the stable public API contract
 * defined in Point 6, 7, and 8 of `IP-SAKTI Frontend and API Integration Brief.md`.
 */

export function normalizeAnswered({ question, answerText, confidence, citations = [], cautions = [], meta = {} }) {
  return {
    success: true,
    status: "answered",
    question: question || "",
    answer: {
      text: answerText || "",
      confidence: confidence || "MODERATE",
    },
    citations: citations.map((c, index) => ({
      id: c.id ?? index + 1,
      marker: c.marker || `[${index + 1}]`,
      document: c.document || c.document_name || "Statutory Document",
      locator: c.locator || c.citation || "Section/Rule",
      page: typeof c.page === "number" ? c.page : (Array.isArray(c.pages) && c.pages[0]) || 1,
      source_status: c.source_status || c.temporal_status || "CURRENT",
      provenance_level: c.provenance_level || c.source_evidence_level || "B",
      evidence_text: c.evidence_text || c.text || "",
    })),
    cautions: Array.isArray(cautions) ? cautions : [],
    meta: {
      answer_status: "PUBLISHED",
      ...meta,
    },
  };
}

export function normalizeAbstained({ question, reason = "The available evidence was insufficient to publish a grounded answer.", cautions = [], meta = {} }) {
  return {
    success: true,
    status: "abstained",
    question: question || "",
    answer: {
      text: "I couldn't produce a sufficiently supported answer from the available legal evidence.",
      confidence: null,
    },
    citations: [],
    cautions: [reason, ...cautions].filter(Boolean),
    meta: {
      answer_status: "ABSTAINED",
      ...meta,
    },
  };
}

export function normalizeError({ message = "The IP-SAKTI service could not process the request.", errorCode = "PIPELINE_ERROR" } = {}) {
  return {
    success: false,
    status: "error",
    message,
    error_code: errorCode,
  };
}

export function normalizeInvalidRequest(message = "Please provide a valid legal question.") {
  return {
    success: false,
    status: "invalid_request",
    message,
  };
}
