"""Phase E: one clarifying question, at most, and only when it earns its place.

The behaviour this replaces is the one a demo audience notices immediately: a
user types "How can I trademark my own firm?", and the system either retrieves
five passages that share a word with the query and answers from whichever looked
closest, or it abstains and tells them to rephrase — and then tells them to
rephrase again. Both are failures, and they are opposite failures, so the fix has
to be careful not to trade one for the other.

Three rules, and every one of them is a bound rather than a behaviour:

**Infer before asking.** A query that names a regime has already chosen its
body of law, and asking "did you mean trade marks?" of someone who wrote
"trademark" is an insult dressed as diligence. `assess` only reaches for a
clarification when the deterministic signals — the Phase 5 router's regime
lexicon, its document aliases, its identifier parser — find *nothing* to anchor
on. That is a property of the query string, computed the same way every time,
with no model and no network.

**Never in place of an answer.** A clarification can be offered in exactly two
places: before retrieval, for a query that anchors on nothing at all, or beside
an abstention the pipeline had already decided on. It can never convert an
answer into a question, and it can never convert an abstention into an answer —
`ABSTAINED` stays `ABSTAINED`, with the same reasons attached, and the
clarification rides alongside as a suggestion. The abstention logic is not
consulted, not softened and not reached by this module.

**At most one, and the caller controls it.** `allow_clarification=False` turns
the whole module off for a request, and the API sets that from the client's own
`clarified` flag. A second pass over the same question therefore runs the full
pipeline whatever it looks like, so no loop can form: the worst case is one
clarifying question followed by a normal answer or a normal abstention.

What it deliberately does not do
--------------------------------

It does not rewrite the query, expand it with synonyms, or send it to a model to
be reformulated. A rewritten query is a query the user did not ask, and when the
answer is wrong nobody can tell whether the pipeline or the rewrite was at
fault. The question that reaches BM25 and BGE-M3 is the one that was typed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ai.answering.policy import content_terms
from ai.retrieval.identifiers import match_documents, parse_identifiers
from ai.retrieval.router import REGIME_TERMS, matched_regimes

CLARIFY_VERSION = "0.1.0"

#: Below this many content terms, with nothing else to anchor on, a query is a
#: topic rather than a question — "patent fees", "biodiversity". Three is chosen
#: because two content words plus a regime term is already a real question and
#: must not be interrupted.
MIN_CONTENT_TERMS = 3

#: One clarifying question per regime, written out rather than composed, so the
#: wording is reviewable and cannot come out mangled for an unusual query. Each
#: names the regime in the user's vocabulary and offers the commonest reading —
#: a question they can answer with "yes" and move on.
REGIME_QUESTIONS: dict[str, str] = {
    "patents": "Do you mean patent protection for an invention in India — "
               "filing, examination, or what can be patented?",
    "trademarks": "Do you mean registering a trade mark in India for your firm's "
                  "name or logo?",
    "geographical_indications": "Do you mean registering or using a geographical "
                                "indication in India?",
    "designs": "Do you mean registering the visual design of an article in India?",
    "copyright": "Do you mean copyright in a work — ownership, term, or "
                 "permitted use?",
    "biodiversity_abs": "Do you mean access to biological resources and "
                        "benefit-sharing under the Biological Diversity Act?",
    "plant_varieties": "Do you mean registering a plant variety or a breeder's "
                       "rights in India?",
    "drug_regulation": "Do you mean licensing or standards for drugs and "
                       "cosmetics in India?",
    "international_ip": "Do you mean the position under the TRIPS Agreement?",
    "ip_policy": "Do you mean the National IPR Policy rather than a specific statute?",
}

#: The regimes offered when a query names none. The corpus's own areas, in the
#: order a reader is likeliest to want them.
REGIME_CHOICES = ("patents", "trademarks", "copyright", "designs",
                  "geographical_indications", "plant_varieties",
                  "biodiversity_abs", "drug_regulation")

REGIME_LABELS = {
    "patents": "patents", "trademarks": "trade marks", "copyright": "copyright",
    "designs": "designs", "geographical_indications": "geographical indications",
    "plant_varieties": "plant varieties", "biodiversity_abs": "biodiversity and "
    "benefit-sharing", "drug_regulation": "drugs and cosmetics",
    "international_ip": "international IP (TRIPS)", "ip_policy": "IP policy",
}

_UNANCHORED = ("no regime term, no instrument name and no provision number were "
               "found in the question, so there is nothing to retrieve against")


@dataclass(frozen=True)
class Clarification:
    """One question to put back to the user, and why it was asked."""

    question: str
    reason: str
    stage: str
    regimes: tuple[str, ...] = ()
    options: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "reason": self.reason,
            "stage": self.stage,
            "regimes": list(self.regimes),
            "options": list(self.options),
            "max_per_request": 1,
            "version": CLARIFY_VERSION,
        }


def anchors(query: str, aliases: Mapping[str, Sequence[str]] | None = None
            ) -> dict[str, Any]:
    """Everything deterministic the query gives us to retrieve against.

    Reuses the Phase 5 router's own lexicon and the Phase 4 identifier parser
    rather than introducing a second opinion about what a query names. If the
    router would route it, this module can anchor on it.
    """
    regimes = matched_regimes(query)
    documents = sorted(match_documents(query, dict(aliases))) if aliases else []
    refs = parse_identifiers(query)
    terms = content_terms(query)
    return {
        "regimes": regimes,
        "documents": documents,
        "identifiers": [f"{r.kind} {r.number}" for r in refs],
        "content_terms": sorted(terms),
        "anchored": bool(regimes or documents or refs),
    }


def pre_retrieval(query: str, aliases: Mapping[str, Sequence[str]] | None = None
                  ) -> Clarification | None:
    """A clarification for a query that names no body of law at all.

    This is the only place the module runs *instead of* the pipeline, and the
    condition is deliberately severe: not "vague", but anchored on nothing the
    router can see and shorter than a real question. "How can I protect my
    idea?" lands here. "How can I trademark my own firm?" does not — it names a
    regime, so it is inferred and answered.
    """
    signals = anchors(query, aliases)
    if signals["anchored"]:
        return None
    if len(signals["content_terms"]) >= MIN_CONTENT_TERMS:
        return None
    options = tuple(REGIME_LABELS[r] for r in REGIME_CHOICES)
    return Clarification(
        question="Which area of IP or regulatory law does your question concern? "
                 "For example: " + ", ".join(options[:4]) + ".",
        reason=_UNANCHORED,
        stage="pre_retrieval",
        options=options)


def post_abstention(query: str, result: Mapping[str, Any],
                    aliases: Mapping[str, Sequence[str]] | None = None
                    ) -> Clarification | None:
    """A suggestion to attach to an abstention the pipeline already made.

    Called only when `status == "ABSTAINED"`. It reads the abstention; it never
    influences it. Where the query named one regime, the clarification offers
    the commonest reading of that regime, which is the question most likely to
    turn a dead end into a retrievable one. Where it named several, it asks
    which — an answer spanning two regimes is the case the evidence policy is
    least able to resolve on its own.
    """
    if result.get("status") != "ABSTAINED":
        return None
    signals = anchors(query, aliases)
    regimes = signals["regimes"]
    if len(regimes) == 1 and regimes[0] in REGIME_QUESTIONS:
        return Clarification(
            question=REGIME_QUESTIONS[regimes[0]],
            reason="the retrieved passages did not support an answer; confirming "
                   "the intended question narrows the search",
            stage="post_abstention",
            regimes=tuple(regimes))
    if len(regimes) > 1:
        labels = [REGIME_LABELS.get(r, r) for r in regimes]
        return Clarification(
            question="Your question touches " + " and ".join(labels)
                     + ". Which of those did you mean?",
            reason="the question spans more than one body of law, and the "
                   "evidence did not settle which",
            stage="post_abstention",
            regimes=tuple(regimes),
            options=tuple(labels))
    if not signals["anchored"]:
        options = tuple(REGIME_LABELS[r] for r in REGIME_CHOICES)
        return Clarification(
            question="Which area of IP or regulatory law does your question "
                     "concern? For example: " + ", ".join(options[:4]) + ".",
            reason=_UNANCHORED,
            stage="post_abstention",
            options=options)
    return None


def clarify_config() -> dict[str, Any]:
    return {
        "version": CLARIFY_VERSION,
        "max_clarifications_per_request": 1,
        "deterministic": True,
        "uses_model": False,
        "rewrites_query": False,
        "min_content_terms": MIN_CONTENT_TERMS,
        "stages": ("pre_retrieval", "post_abstention"),
        "regimes_covered": sorted(REGIME_TERMS),
        "guarantee": "a clarification never changes a status: it is returned "
                     "instead of running the pipeline only for a query that "
                     "anchors on nothing, and is otherwise attached to an "
                     "abstention the pipeline had already decided",
    }
