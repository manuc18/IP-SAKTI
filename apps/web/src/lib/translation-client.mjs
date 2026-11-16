function addString(values, value) {
  if (typeof value === "string" && value.trim()) values.add(value);
}

function addCitation(values, citation) {
  if (!citation || typeof citation !== "object") return;
  addString(values, citation.citation_text);
  addString(values, citation.citation);
  addString(values, citation.document);
  addString(values, citation.document_name);
  addString(values, citation.locator);
  addString(values, citation.locator_text);
  addString(values, citation.official_source_url);
  addString(values, citation.source_document?.official_source_url);
}

export function collectProtectedTerms(result) {
  const values = new Set();
  const answerCitations = result?.answer?.citations || [];
  const topLevelCitations = result?.citations || [];
  const evidence = result?.evidence || [];
  const sources = result?.sources || [];

  for (const item of [
    ...answerCitations,
    ...topLevelCitations,
    ...evidence,
    ...sources,
  ]) {
    addCitation(values, item);
  }

  return [...values].sort((left, right) => right.length - left.length);
}
