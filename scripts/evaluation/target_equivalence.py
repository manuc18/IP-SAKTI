"""Which other chunks answer a generated question as well as its label does.

`scripts/evaluation/generate_synthetic_benchmark.py` writes exactly one target
chunk per question, because it builds each question *from* one chunk and never
looks at another. That is fine for provenance — every span it claims is verbatim
in the chunk it names, and `verify()` refuses to write the file otherwise — and
it is wrong for scoring, because the corpus contains many provisions that answer
the same question equally well.

Three measured examples, all from the rank-1 failures:

* `IN-DC-RULES-1945::part-vi-rule-67d` and `...rule-62` are both headed "Sale at
  more than one place" with near-identical text. Which one the generator happened
  to build from is a coin toss, and retrieval is scored as wrong for calling it
  the other way.
* `IN-PATENTS-ACT-1970::chapter-v-section-25-c1` and `-c2` are the two halves of
  section 25 as the chunker split it. A question about "opposition to the patent"
  cannot distinguish them, and the deadline it asks about is in `-c2` while the
  label is `-c1`.
* 32 of the 59 PPVFR-Rules questions are built from headings like "Fee for
  conducting tests **under section 19**", so the question names a section of the
  *Act* while the label is a rule of the *Rules*. Those score 3/32.

So this module computes, for each question, the set of chunks that would be a
correct answer — from the question text and the corpus, never from the run being
scored. Three rules, and each is a claim that can be argued with:

**R1, same provision.** A chunk with the same *locator* as the target — same
document, same section or rule number — ignoring the subsection and clause
fields. The chunker split one provision into parts; the question was written
about the provision, and `Section 25` and `Section 25(2)` are two chunks of it.
Matching on the citation string instead would miss exactly that case, because the
two parts carry different citation strings.

**R2, same heading, only when the question names no instrument.** "May the
authority act of its own motion in respect of security for costs?" names no Act,
and "Security for costs" is the heading of both Trade Marks Rule 51 and Patents
Act section 150. Both answer it. But when the question *does* name its
instrument, a same-heading chunk from a different one is a genuine miss and stays
one — the guard is the same `match_documents` the retriever uses.

**R3, a provision the question names, in the same body of law.** If the question
says "under section 19", section 19 is an acceptable answer to it — but only the
one in the regime the question is about. Unscoped, this rule accepted section 19
of all eight Acts that have one, which would turn a hard question into a
gimme and make the benchmark flatter itself. The scope is the target's own
`domain` plus any document the question names, so the PPVFR Rules question
reaches the PPVFR Act and stops there.

Parsed with `parse_identifiers`, the same parser retrieval uses.

Nothing here is a per-question exception list and nothing reads the run's
output. Both the strict and the equivalence-adjusted figures are reported, always
together: the strict number is what a single-label benchmark measures, and the
adjusted number is what it was trying to measure.
"""
from __future__ import annotations

import collections
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

from ai.retrieval.identifiers import (  # noqa: E402
    document_aliases,
    match_documents,
    parse_identifiers,
)

EQUIVALENCE_VERSION = "0.1.0"

_PUNCT = re.compile(r"[^a-z0-9 ]+")

#: Locator fields a parsed identifier can name, mapped to the parser's own kind.
_KINDS = ("section", "rule", "article", "regulation", "paragraph", "form",
          "schedule", "chapter", "part")


def normalise_heading(heading: str | None) -> str:
    """Case, punctuation and spacing folded; wording untouched.

    Deliberately crude. Two headings are "the same heading" when they are the
    same words — not when they are similar, because a similarity threshold here
    would silently decide how generous the benchmark is.
    """
    text = (heading or "").lower().replace("’", "").replace("'", "")
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def _normalise_number(value: Any) -> str:
    """`3`, `3.`, `03` and `3A` compared the way the locator parser writes them."""
    return str(value).strip().strip(".").lstrip("0").lower() or "0"


def _locator_key(document_id: str, locator: Mapping[str, Any] | None) -> str | None:
    """A provision's identity within its document, or None if it has no locator."""
    if not locator:
        return None
    parts = [f"{field}={locator.get(field)}" for field in _KINDS
             if locator.get(field) not in (None, "")]
    return f"{document_id}|" + "|".join(parts) if parts else None


@dataclass
class EquivalenceIndex:
    """The corpus, indexed the three ways the rules above need it."""

    by_citation: dict[str, list[str]] = field(default_factory=dict)
    by_heading: dict[str, list[str]] = field(default_factory=dict)
    by_locator: dict[str, list[str]] = field(default_factory=dict)
    #: (document_id, kind, number) -> chunks, for R3. Keyed on the parsed
    #: fields rather than on a substring of the locator key: matching
    #: `"section=3" in "IN-DC-ACT-1940|chapter=I|section=30"` is true, and a
    #: first version of R3 therefore accepted sections 3, 30, 33 and 3A for a
    #: question about section 3 — 36 extra chunks on one question.
    by_provision: dict[tuple[str, str, str], list[str]] = field(default_factory=dict)
    chunk: dict[str, dict[str, Any]] = field(default_factory=dict)
    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def build(cls, chunks: Iterable[Any]) -> "EquivalenceIndex":
        index = cls()
        materialised = []
        for chunk in chunks:
            record = chunk if isinstance(chunk, dict) else _as_record(chunk)
            materialised.append(record)
            chunk_id = record["chunk_id"]
            index.chunk[chunk_id] = record

            citation = (record.get("citation") or "").strip()
            if citation:
                key = f"{record['document_id']}|{citation}"
                index.by_citation.setdefault(key, []).append(chunk_id)

            heading = normalise_heading(record.get("heading"))
            if heading:
                index.by_heading.setdefault(heading, []).append(chunk_id)

            locator = record.get("locator") or {}
            locator_key = _locator_key(record["document_id"], locator)
            if locator_key:
                index.by_locator.setdefault(locator_key, []).append(chunk_id)
            for kind in _KINDS:
                number = locator.get(kind)
                if number in (None, ""):
                    continue
                index.by_provision.setdefault(
                    (record["document_id"], kind, _normalise_number(number)),
                    []).append(chunk_id)

        index.aliases = document_aliases(
            [_AliasView(record) for record in materialised])
        return index

    # -- the one public operation --------------------------------------------

    def acceptable(self, question: str,
                   target_chunk_ids: Sequence[str]) -> tuple[set[str], dict[str, int]]:
        """Every chunk that answers `question` as well as its labels do.

        Returns the set — which always contains the labels themselves — and a
        count per rule, so a summary can say where the widening came from
        instead of only reporting that it happened.
        """
        accepted = set(target_chunk_ids)
        counts = {"R1_same_provision": 0, "R2_same_heading": 0,
                  "R3_named_provision": 0}

        for chunk_id in list(target_chunk_ids):
            record = self.chunk.get(chunk_id)
            if record is None:
                continue
            locator_key = _locator_key(record["document_id"], record.get("locator"))
            if locator_key:
                new = set(self.by_locator.get(locator_key, ())) - accepted
                counts["R1_same_provision"] += len(new)
                accepted |= new

        # R2 applies only where the question left the instrument open.
        if not match_documents(question, self.aliases):
            for chunk_id in list(target_chunk_ids):
                record = self.chunk.get(chunk_id)
                if record is None:
                    continue
                heading = normalise_heading(record.get("heading"))
                if not heading:
                    continue
                new = set(self.by_heading.get(heading, ())) - accepted
                counts["R2_same_heading"] += len(new)
                accepted |= new

        refs = parse_identifiers(question)
        if refs:
            scope = self._scope(question, target_chunk_ids)
            for ref in refs:
                number = _normalise_number(ref.number)
                for document_id in scope:
                    ids = self.by_provision.get((document_id, ref.kind, number))
                    if not ids:
                        continue
                    new = set(ids) - accepted
                    counts["R3_named_provision"] += len(new)
                    accepted |= new

        return accepted, counts

    def _scope(self, question: str, target_chunk_ids: Sequence[str]) -> set[str]:
        """The documents R3 may reach: the target's own body of law, and any named.

        "Under section 19" in a question about the PPVFR Rules means section 19
        of the PPVFR Act, not of the Copyright Act. Domain is the chunker's own
        regime label, so this is the corpus's opinion about which instruments
        belong together rather than a second one invented here.
        """
        scope = set(match_documents(question, self.aliases))
        domains = set()
        for chunk_id in target_chunk_ids:
            record = self.chunk.get(chunk_id)
            if record is None:
                continue
            scope.add(record["document_id"])
            if record.get("domain"):
                domains.add(record["domain"])
        if domains:
            scope |= {record["document_id"] for record in self.chunk.values()
                      if record.get("domain") in domains}
        return scope


@dataclass
class _AliasView:
    """`document_aliases` reads attributes; the chunk records are dicts."""

    record: Mapping[str, Any]

    @property
    def document_id(self) -> str:
        return self.record["document_id"]

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.record

    def __getattr__(self, name: str) -> Any:
        return self.record.get(name)


#: Fields read off a chunk, in the two places they can live. `ai.retrieval.base.Chunk`
#: exposes them directly; a `RetrievalResult` carries them in `metadata`. Reading
#: only the second silently produced an index with no headings and no locators,
#: and therefore an equivalence set that never widened anything — a null result
#: that looked like a finding.
_FIELDS = ("citation", "heading", "domain", "short_title", "title")


def _as_record(chunk: Any) -> dict[str, Any]:
    metadata = dict(getattr(chunk, "metadata", {}) or {})
    record: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
    }
    for name in _FIELDS:
        record[name] = getattr(chunk, name, None) or metadata.get(name)
    record["locator"] = (getattr(chunk, "locator", None)
                         or metadata.get("locator_fields")
                         or metadata.get("locator"))
    return record


def equivalence_config() -> dict[str, Any]:
    return {
        "version": EQUIVALENCE_VERSION,
        "rules": {
            "R1_same_provision": "another chunk of the same provision in the "
                                 "same document (identical citation)",
            "R2_same_heading": "a chunk with the same normalised heading, "
                               "applied only when the question names no "
                               "instrument",
            "R3_named_provision": "a chunk whose locator matches a provision "
                                  "the question itself names, within the "
                                  "target's own regime or a document the "
                                  "question names",
        },
        "derived_from": "the question text and the corpus only; never from the "
                        "run being scored",
        "reporting": "strict and adjusted figures are always reported together",
    }
