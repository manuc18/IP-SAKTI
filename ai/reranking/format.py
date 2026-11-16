"""The one canonical way a chunk is written down for a reranker to read.

Defined once, in one place, because the alternative is two rerankers being
compared on two different inputs and the difference being reported as a model
result. Phase 6 §9 asks for exactly this.

The format is three or four labelled lines:

    Document: The Patents Act, 1970
    Locator: Section 3
    Heading: What are not inventions
    Text: (a) a discovery of a scientific principle ...

`Heading` is omitted when the chunk has none rather than written as empty, so a
model never sees a label with nothing after it.

**What is deliberately included.** `Document` and `Locator` are the two things a
lawyer reads before the text itself, and a cross-encoder asked "what does
Section 3(p) say" has no way to prefer the right provision if the provision
number is not in the passage — the *text* of section 3 says "3." on one line and
"(p)" thirty lines later, which is the asymmetry `tokenizer.py` already
documents. Withholding the locator would not make the evaluation purer; it would
make it measure a different task.

**What is deliberately excluded, and why it is tested.** The representation must
carry nothing that identifies a chunk as the benchmark's answer:

* `chunk_id` — not a fact about the document. It is the join key the golden
  dataset stores its labels under, and a model that learned to read it would be
  reading the answer key. It is also the one field an evaluation harness is most
  likely to paste in "for debugging".
* `citable`, `source_evidence_level`, `temporal_status`, `source_status` — real
  facts, but facts about *our evidence for* the document rather than about what
  it says. Phase 6 §6 requires relevance, authority and citability to stay
  separable; folding authority into the text the relevance model reads destroys
  that separation at the first layer. They travel in the candidate's metadata,
  where an explicit policy can act on them.
* anything from the golden record — expected chunks, expected locators, anchor
  terms, notes, the query family label, `verified`.

:func:`leakage_report` checks a rendered pool against all of this and is run by
the benchmark on every experiment, not only by the tests.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from ai.retrieval.candidates import Candidate

#: Fields of the chunk record that may appear in a rendered candidate. Anything
#: outside this set is either irrelevant to relevance or is evidence-about-
#: evidence that belongs to the authority policy instead.
ALLOWED_FIELDS = ("document", "locator", "heading", "text")

#: Metadata keys a rendered candidate must never contain, checked by name and by
#: value. `chunk_id` is checked separately because it is not in `metadata`.
FORBIDDEN_METADATA = (
    "citability_reason",
    "source_evidence_level",
    "source_status",
    "temporal_status",
    "locator_unavailable",
    "retrieval_path",
    "query_type",
    "routed_to",
    "routing_reason",
)


def document_name(candidate: Candidate) -> str:
    """How the instrument is named, preferring the short title a reader uses."""
    metadata = candidate.metadata
    return str(metadata.get("short_title") or metadata.get("title")
               or candidate.document_id)


def locator_line(candidate: Candidate) -> str:
    """The chunk's own citation string, minus the document name it repeats.

    `citation` reads "Patents Act, 1970 — Section 3"; the document is already on
    its own line, so repeating it wastes the sequence budget the cross-encoder
    has to spend on operative text.

    The em dash is Phase 2's separator and every citation in the corpus carries
    one, so splitting on it is exact. The prefix fallback is there for a
    hand-built citation that does not — stripping a known document name is still
    better than repeating it, and a citation matching neither is passed through
    whole rather than mangled.
    """
    citation = (candidate.locator or "").strip()
    name = document_name(candidate)
    head, separator, tail = citation.partition(" — ")
    if separator:
        citation = tail
    elif citation.startswith(name):
        citation = citation[len(name):]
    return citation.lstrip(" —–-,").strip() or "(no locator established)"


def format_candidate(candidate: Candidate, *, text_budget: int | None = None) -> str:
    """Render one candidate. Deterministic: same chunk, same string, always.

    `text_budget` truncates the operative text to a character count. It defaults
    to ``None`` — no truncation — so that any shortening a model does is the
    model's own tokenizer truncation, which :mod:`ai.reranking.cross_encoder`
    counts and reports. Pre-truncating here would hide that count.
    """
    lines = [f"Document: {document_name(candidate)}",
             f"Locator: {locator_line(candidate)}"]
    heading = re.sub(r"\s+", " ", (candidate.metadata.get("heading") or "")).strip()
    if heading:
        lines.append(f"Heading: {heading}")
    text = re.sub(r"[ \t]+", " ", candidate.text).strip()
    if text_budget is not None and len(text) > text_budget:
        text = text[:text_budget].rstrip()
    lines.append(f"Text: {text}")
    return "\n".join(lines)


def format_config(text_budget: int | None = None) -> dict[str, Any]:
    return {
        "representation": "labelled lines: Document / Locator / [Heading] / Text",
        "fields": list(ALLOWED_FIELDS),
        "heading_omitted_when_absent": True,
        "text_budget_chars": text_budget,
        "whitespace": "runs of spaces and tabs collapsed; line breaks preserved",
        "excluded": ["chunk_id", "citable", "source_evidence_level",
                     "temporal_status", "source_status", "physical_pages",
                     "retrieval scores", "retrieval path", "golden labels"],
    }


def leakage_report(query: str, candidates: Sequence[Candidate], *,
                   rendered: Iterable[str] | None = None,
                   forbidden_strings: Mapping[str, Sequence[str]] | None = None,
                   text_budget: int | None = None) -> dict[str, Any]:
    """Audit what the reranker can actually see. Phase 6 §18.

    `forbidden_strings` maps a label to strings that must not appear in any
    rendered candidate *or* in the query — the golden record's expected chunk
    ids, its expected document ids, its private notes. The query is checked too,
    because a query with the answer key pasted into it would leak just as
    thoroughly as a candidate that had.

    Two things are deliberately *not* forbidden, and saying why matters as much
    as the checks themselves. An expected **locator** may legitimately appear:
    the query is allowed to say "Section 3(p)", which Phase 6 §18 permits in
    terms, and the chunk's own citation is part of the canonical representation
    for the reason `format_candidate` gives. **Anchor terms** are validated by
    `validate_golden_dataset.py` to occur in the expected chunk's text, so
    forbidding them would forbid the corpus from containing itself; they are
    evidence the ground truth was derived from the corpus, not a label about it.
    """
    texts = list(rendered) if rendered is not None else [
        format_candidate(c, text_budget=text_budget) for c in candidates]
    findings: list[dict[str, Any]] = []

    for candidate, text in zip(candidates, texts):
        if candidate.chunk_id in text:
            findings.append({"check": "chunk_id_in_rendered_candidate",
                             "chunk_id": candidate.chunk_id})
        for key in FORBIDDEN_METADATA:
            value = candidate.metadata.get(key)
            if isinstance(value, str) and value and f"{key}: {value}" in text:
                findings.append({"check": "forbidden_metadata_rendered",
                                 "chunk_id": candidate.chunk_id, "field": key})
        # Only the header is audited for stray labels. Everything from `Text:`
        # onward is the instrument's own prose, which contains lines like
        # "Provided that:" that are not labels and must not be read as any.
        header = text.split("\nText:", 1)[0]
        stray = [line.split(":", 1)[0].strip().lower()
                 for line in header.split("\n") if ":" in line]
        unknown = sorted({label for label in stray if label not in ALLOWED_FIELDS})
        if unknown:
            findings.append({"check": "unknown_label_in_rendered_candidate",
                             "chunk_id": candidate.chunk_id, "labels": unknown})

    for label, strings in (forbidden_strings or {}).items():
        for needle in strings:
            if not needle:
                continue
            if needle in query:
                findings.append({"check": "forbidden_string_in_query",
                                 "source": label, "value": needle})
            hit = next((c.chunk_id for c, t in zip(candidates, texts) if needle in t), None)
            if hit is not None:
                findings.append({"check": "forbidden_string_in_rendered_candidate",
                                 "source": label, "value": needle, "chunk_id": hit})

    return {
        "candidates_audited": len(texts),
        "clean": not findings,
        "findings": findings,
    }
