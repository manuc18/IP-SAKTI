"""Deterministic query routing — a policy layer, not a retrieval engine.

Phase 4 measured two retrievers that fail in opposite directions. BM25 with an
identifier boost put every cited provision in the top five (Recall@5 1.000 on
identifier queries) and was worth almost nothing on the rest; BGE-M3 owned the
conceptual queries and was *worse* than the boost on cited ones. Averaging the
two, whether by RRF or anything else, spends one family's strength paying for
the other's weakness. Choosing between them costs nothing and needs no training.

So this module decides which existing retriever answers a query. It does not
rank, score or index anything, and it never edits a result — a routed result is
the chosen retriever's result with the route recorded on it.

The classification is deterministic and explainable by construction: every
decision names the evidence that produced it, and the same query always routes
the same way. No LLM, no trained classifier, no embeddings. `confidence` is the
literal string ``"deterministic"`` rather than a number, because a number here
would be invented — there is no probability model behind a regex.

Three families:

``IDENTIFIER``   The query names a provision. :func:`parse_identifiers` finds it.
``CONCEPTUAL``   No provision named, but the query names the regime it is asking
                 about ("patentable", "benefit sharing"), so retrieval starts
                 with the body of law already narrowed.
``GENERAL``      An open practical question ("How do I protect an Ayurvedic
                 formulation?") that names neither a provision nor a regime, so
                 retrieval has to discover the regime itself.

The CONCEPTUAL/GENERAL split is **diagnostic only**: Phase 5 routes both to
BGE-M3, so nothing in the ranking depends on it. It is reported separately
because a router that cannot say how often it met a question with no legal
anchor at all cannot tell us what to build next. The test that separates them is
a surface-form one — a procedural opener, then a regime lexicon — and it is
honest about being surface-form; see `docs/RETRIEVAL_ROUTING.md`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ai.retrieval.base import Filters, RetrievalResult, Retriever
from ai.retrieval.identifiers import LocatorRef, match_documents, parse_identifiers

IDENTIFIER = "IDENTIFIER"
CONCEPTUAL = "CONCEPTUAL"
GENERAL = "GENERAL"
QUERY_TYPES = (IDENTIFIER, CONCEPTUAL, GENERAL)

#: Retrieval paths a routed answer can arrive by. Every result records exactly
#: one of these, including the fallbacks — a fallback that is not visible in the
#: output is a silent change of engine, and the failure it causes gets blamed on
#: the retriever the caller thought was running.
PATHS = (
    "identifier",
    "identifier_fallback_dense",
    "conceptual_dense",
    "conceptual_fallback_lexical",
    "general_dense",
    "general_fallback_lexical",
)

#: A query that opens this way is asking how to do something rather than what
#: the law provides. "How do I protect an Ayurvedic formulation?" spans patents,
#: GI and traditional-knowledge regimes at once; the phrasing is the evidence
#: that the asker has not picked one, not a guess about what they meant.
PROCEDURAL_OPENER = re.compile(
    r"(?i)^\s*(?:so\s+)?(?:how\s+(?:do|can|would|should|might)\s+(?:i|we|you|one)\b"
    r"|how\s+to\b"
    r"|how\s+does\s+(?:one|someone)\b"
    r"|what\s+(?:steps|is\s+the\s+process|is\s+the\s+procedure)\b"
    r"|where\s+(?:do|can|should)\s+(?:i|we)\b"
    r"|who\s+(?:do|should)\s+(?:i|we)\b"
    r"|can\s+(?:i|we)\s+(?:apply|register|file|get|obtain)\b"
    r"|steps\s+to\b)"
)

#: Terms that name a body of law rather than a subject matter. Matching one says
#: the query has already chosen its regime, which is what separates CONCEPTUAL
#: from GENERAL. Subject-matter words are deliberately absent: "Ayurvedic" names
#: a thing that several regimes regulate, so it settles nothing.
REGIME_TERMS: dict[str, tuple[str, ...]] = {
    "patents": ("patent", "patentab", "patentee", "prior art", "compulsory licence",
                "compulsory license", "invention", "specification", "priority date",
                "working of patent", "revocation"),
    "trademarks": ("trade mark", "trademark", "passing off", "well-known mark",
                   "deceptively similar", "collective mark", "certification mark"),
    "geographical_indications": ("geographical indication", "authorised user",
                                 "registered proprietor of a gi", "homonymous"),
    "designs": ("registered design", "design registration", "piracy of a design",
                "industrial design"),
    "copyright": ("copyright", "fair dealing", "moral right", "neighbouring right",
                  "performer's right", "broadcast reproduction"),
    "biodiversity_abs": ("biological resource", "biological diversity", "biodiversity",
                         "benefit sharing", "benefit-sharing", "access and benefit",
                         "prior informed consent", "mutually agreed terms",
                         "genetic resource", "traditional knowledge associated"),
    "plant_varieties": ("plant variety", "plant varieties", "breeder", "farmers' right",
                        "farmers right", "farm-saved", "essentially derived",
                        "denomination", "gene fund"),
    "drug_regulation": ("misbranded", "adulterated drug", "spurious drug",
                        "loan licence", "manufacture for sale of drugs"),
    "international_ip": ("trips", "national treatment", "most-favoured-nation",
                         "wto member"),
    "ip_policy": ("ipr policy", "national ipr"),
}


@dataclass(frozen=True)
class RoutingDecision:
    """Why a query was routed where it was. The `reason` is the audit trail."""

    query_type: str
    confidence: str
    matched_identifiers: tuple[str, ...]
    reason: str
    matched_documents: tuple[str, ...] = ()
    matched_regimes: tuple[str, ...] = ()
    refs: tuple[LocatorRef, ...] = field(default=(), repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "confidence": self.confidence,
            "matched_identifiers": list(self.matched_identifiers),
            "reason": self.reason,
            "matched_documents": list(self.matched_documents),
            "matched_regimes": list(self.matched_regimes),
        }


def _format_ref(ref: LocatorRef) -> str:
    tail = "".join(f"({part})" for part in ref.subdivisions)
    return f"{ref.kind} {ref.number}{tail}"


def matched_regimes(query: str) -> list[str]:
    """Regimes the query names outright, in a stable order."""
    lowered = re.sub(r"\s+", " ", query.lower())
    return sorted(regime for regime, terms in REGIME_TERMS.items()
                  if any(term in lowered for term in terms))


def classify(query: str,
             aliases: Mapping[str, Sequence[str]] | None = None) -> RoutingDecision:
    """Route a query. Same input, same output, always — and it says why.

    The order of the tests is the substance of the policy. A named provision
    wins outright: it is the least ambiguous thing a legal query can contain,
    and no amount of surrounding prose makes it less so. Failing that, a
    procedural opener is checked *before* the regime lexicon, because
    "How do I register a trade mark?" names a regime but is still a request for
    a procedure rather than a question about doctrine.
    """
    refs = parse_identifiers(query)
    documents = sorted(match_documents(query, dict(aliases))) if aliases else []
    regimes = matched_regimes(query)

    if refs:
        named = tuple(_format_ref(ref) for ref in refs)
        where = f" of {', '.join(documents)}" if documents else ""
        return RoutingDecision(
            query_type=IDENTIFIER,
            confidence="deterministic",
            matched_identifiers=named,
            reason=f"the query names {', '.join(named)}{where}; an exact locator "
                   f"match is decidable from chunk metadata and does not need "
                   f"semantic similarity",
            matched_documents=tuple(documents),
            matched_regimes=tuple(regimes),
            refs=tuple(refs),
        )

    opener = PROCEDURAL_OPENER.match(query)
    if opener:
        return RoutingDecision(
            query_type=GENERAL,
            confidence="deterministic",
            matched_identifiers=(),
            reason=f"no provision named, and the query opens with "
                   f"{opener.group(0).strip()!r} — it asks how to proceed rather "
                   f"than what a regime provides, so the regime itself has to be "
                   f"found by retrieval",
            matched_documents=tuple(documents),
            matched_regimes=tuple(regimes),
        )

    if regimes or documents:
        evidence = ", ".join(regimes or documents)
        return RoutingDecision(
            query_type=CONCEPTUAL,
            confidence="deterministic",
            matched_identifiers=(),
            reason=f"no provision named, but the query states its subject "
                   f"({evidence}); the answer is a passage about that regime, "
                   f"which is a semantic-similarity problem",
            matched_documents=tuple(documents),
            matched_regimes=tuple(regimes),
        )

    return RoutingDecision(
        query_type=GENERAL,
        confidence="deterministic",
        matched_identifiers=(),
        reason="the query names neither a provision nor a regime this lexicon "
               "recognises, so retrieval has to discover the body of law itself",
        matched_documents=tuple(documents),
        matched_regimes=(),
    )


class RoutedRetriever(Retriever):
    """Picks one existing retriever per query. Adds no ranking of its own.

    Phase 5 §6 fixes the policy: IDENTIFIER to the boosted lexical retriever,
    CONCEPTUAL and GENERAL to BGE-M3. RRF is deliberately not in the table —
    Phase 4 showed no fusion beating BGE-M3 at rank 1, so fusing here would give
    up the thing routing exists to protect.
    """

    name = "routed"
    version = "0.1.0"

    def __init__(self, *, identifier: Retriever, dense: Retriever,
                 lexical: Retriever,
                 aliases: Mapping[str, Sequence[str]] | None = None):
        self.identifier = identifier
        self.dense = dense
        self.lexical = lexical
        self.aliases = dict(aliases or {})

    def route(self, query: str) -> RoutingDecision:
        return classify(query, self.aliases)

    @staticmethod
    def _boost_fired(results: Sequence[RetrievalResult]) -> bool:
        """Did the identifier retriever actually match the citation it was given?

        A boost of zero on every hit means no chunk in the corpus carries the
        locator the query named — the ranking that came back is plain BM25 over
        a query whose most informative token matched nothing. That is the
        condition Phase 5 §7 calls "no reasonable candidates", and it is read
        from the retriever's own output rather than guessed from the scores.
        """
        return any(float(r.metadata.get("identifier_boost") or 0.0) > 0.0
                   for r in results)

    def search(self, query: str, filters: Filters | None = None,
               top_k: int = 10) -> list[RetrievalResult]:
        decision = self.route(query)

        if decision.query_type == IDENTIFIER:
            results = self.identifier.search(query, filters, top_k)
            if results and self._boost_fired(results):
                return self._stamp(results, decision, "identifier", self.identifier)
            fallback = self.dense.search(query, filters, top_k)
            if fallback:
                return self._stamp(fallback, decision, "identifier_fallback_dense",
                                   self.dense)
            return self._stamp(results, decision, "identifier", self.identifier)

        prefix = "conceptual" if decision.query_type == CONCEPTUAL else "general"
        results = self.dense.search(query, filters, top_k)
        if results:
            return self._stamp(results, decision, f"{prefix}_dense", self.dense)
        # BGE-M3 returning nothing means the filters emptied the corpus, not that
        # the query was hard; BM25 is asked the same question under the same
        # filters so the caller sees why the list is short.
        lexical = self.lexical.search(query, filters, top_k)
        return self._stamp(lexical, decision, f"{prefix}_fallback_lexical", self.lexical)

    @staticmethod
    def _stamp(results: Sequence[RetrievalResult], decision: RoutingDecision,
               path: str, source: Retriever) -> list[RetrievalResult]:
        """Record the route on each result without touching anything else.

        The chunk's provenance — locator, citability, temporal status, evidence
        level — is carried through exactly as the chosen retriever produced it.
        Routing decides *which* engine answers, never *what* a chunk is.
        """
        stamped = []
        for result in results:
            metadata = dict(result.metadata)
            metadata["retrieval_path"] = path
            metadata["query_type"] = decision.query_type
            metadata["routed_to"] = source.name
            metadata["routing_reason"] = decision.reason
            metadata["matched_identifiers"] = list(decision.matched_identifiers)
            stamped.append(RetrievalResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                score=result.score,
                rank=result.rank,
                text=result.text,
                locator=result.locator,
                citable=result.citable,
                metadata=metadata,
            ))
        return stamped

    def config(self) -> dict[str, Any]:
        return {
            "retriever": self.name,
            "version": self.version,
            "policy": {
                IDENTIFIER: self.identifier.name,
                CONCEPTUAL: self.dense.name,
                GENERAL: self.dense.name,
            },
            "fallbacks": {
                "identifier_no_locator_match": self.dense.name,
                "dense_empty": self.lexical.name,
            },
            "paths": list(PATHS),
            "classifier": "deterministic regex + regime lexicon; no model",
            "regime_terms": sum(len(v) for v in REGIME_TERMS.values()),
            "components": {
                "identifier": self.identifier.config(),
                "dense": self.dense.config(),
                "lexical": self.lexical.config(),
            },
        }
