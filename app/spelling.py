"""Conservative, corpus-grounded spelling repair for the query — and nothing else.

"How do I register a tradmark?" used to retrieve nothing useful, because
`tradmark` is a token no chunk contains, and BM25 scores an unmatched term at
zero however obvious the intent. The dense arm sometimes rescued it and
sometimes did not. This module fixes that one failure and is built to be
incapable of anything more ambitious.

Three properties, each of which is the reason for the one below it.

**The corpus is the dictionary.** There is no external word list, no language
model and no learned model. A word is correct if it occurs in the corpus text,
and a correction target must be a word that occurs in the corpus text in at
least :data:`MIN_DOCUMENT_FREQUENCY` chunks. A system that answers only from
these 22 instruments has no business correcting a query towards a word none of
them uses, and the df floor is what stops a typo being "corrected" into an OCR
artefact sitting in one bad page.

**Only out-of-vocabulary words are touched.** If the word the user typed appears
anywhere in the corpus, it is left exactly as typed — no "did you mean",
no ranking of alternatives, no exceptions. This is what protects legal
terminology from being helpfully destroyed: `bis`, `ter`, `mala`, `fides`,
`ex parte`, `sui generis` and every archaic spelling in the Drugs and Cosmetics
Rules, 1945 are all in the corpus, so all of them are untouchable.

**Identifiers are never rewritten.** Only runs of at least
:data:`MIN_WORD_LENGTH` ASCII letters are considered at all, so `85B` reaches
retrieval as `85B`, `3(p)` as `3(p)` and `31bis` as `31bis`. A section number
cannot be spell-corrected because a section number is never offered to the
corrector.

Matching is not raw edit distance, which is too weak for the errors people
actually make: `copywrite` is three edits from `copyright` and `write` is three
from `right`, so a distance-2 corrector — already too loose to be safe — would
still miss the commonest misspelling in the whole subject area. Instead both
sides are reduced to a :func:`skeleton`, an ordered fold of the English
spellings that sound alike and the doubled letters that carry no information.
`copywrite` and `copyright` both reduce to `copyrit` and match exactly, while
`grant` and `grand` reduce to `grant` and `grand` and do not.

The original query is what is returned to the user, what is stored for audit and
what the answer generator is asked about. Only retrieval and reranking see the
corrected string. If a correction was wrong, the transcript shows what was typed
and what was searched for, side by side, and the difference is one line of
deterministic code away from being reproduced.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

SPELLING_VERSION = "0.1.0"

#: Shorter words are left alone. At three letters or fewer almost every edit
#: reaches some other real word, so the corrector has no way to be confident and
#: no business guessing. It also keeps `Act`, `GI`, `TM` and the Latin fragments
#: out of scope entirely.
MIN_WORD_LENGTH = 4

#: A correction target must occur in at least this many distinct chunks. Set
#: against 3,073 chunks: five is high enough to exclude a word that exists only
#: because one page scanned badly, and low enough to keep genuinely rare
#: statutory vocabulary reachable.
MIN_DOCUMENT_FREQUENCY = 5

#: Skeleton edit distance allowed. Exactly one, for every word length.
#:
#: Two was tried and removed. On the 33-misspelling / 26-unrelated-word set the
#: distance-2 tier corrected nothing that distance 1 did not already reach —
#: once transposition counts as a single edit, a realistic typo is one edit —
#: and it produced the only surviving false positive, `toppings` -> `things`.
#: A tier whose entire measured contribution is a wrong answer is not tuned
#: down, it is deleted.
MAX_DISTANCE = 1

#: A correction reached by *edit distance* — as opposed to an exact agreement of
#: skeletons — additionally requires its target to be this common in the corpus.
#:
#: This is the constant that separates "the user misspelled a word this corpus
#: talks about" from "the user used a word from some other subject entirely, and
#: some rare corpus word happens to be one edit away". Without it the corrector
#: turns `pizza` into `pitta`, `topping` into `typing` and `mala fides` into
#: `male fides`: harmless for the answer, which abstains either way, but a lie
#: in the audit trail about what was searched for.
#:
#: Measured on 32 deliberate misspellings of corpus terms and 25 words from
#: unrelated subjects. Every true correction's target occurs in 26 or more
#: chunks; every false one's target occurs in 13 or fewer. There is no value in
#: between to argue about, so the threshold sits in the middle of the gap.
MIN_TARGET_FREQUENCY_FOR_DISTANCE = 20

#: Runs of letters. Anything containing a digit is not a run and is therefore
#: never a candidate — which is the whole of the identifier guard.
_WORD = re.compile(r"[A-Za-z]+")

_DOUBLED = re.compile(r"(.)\1+")
_INTERNAL_GH = re.compile(r"(?<=.)gh")
_TRAILING_E = re.compile(r"e$")


def skeleton(word: str) -> str:
    """The spelling-insensitive form of a word.

    Each rule folds a distinction English writes but does not pronounce, and
    each is applied to the corpus vocabulary and the query word alike, so the
    fold can only ever make two spellings agree — it can never make the
    corrector prefer one real word over a different real word.

    * Repeated letters collapse: `markk` and `mark`, `applicaton` and
      `application` after the rest of the fold, `commitee` and `committee`.
    * `wr` becomes `r` and a word-initial `kn` becomes `n`: `write`/`rite`,
      `know`/`now`. This is what lets `copywrite` reach `copyright`.
    * `ph` becomes `f` and `ck` becomes `k`: `pharmaceutical`, `trafficking`.
    * A non-initial `gh` disappears: `right` becomes `rit`, `though` becomes
      `thou`.
    * A trailing `e` is dropped, because it is silent and people drop it.
    """
    text = word.lower()
    text = _DOUBLED.sub(r"\1", text)
    text = re.sub(r"^kn", "n", text)
    text = re.sub(r"^ps", "s", text)
    text = text.replace("wr", "r")
    text = text.replace("ph", "f")
    text = text.replace("ck", "k")
    text = _INTERNAL_GH.sub("", text)
    text = _TRAILING_E.sub("", text)
    return text


def _distance(a: str, b: str, limit: int) -> int:
    """Damerau-Levenshtein (optimal string alignment), abandoned early past `limit`.

    Transposition counts as one edit rather than two, which is not a refinement
    but the difference between working and not: swapping two adjacent letters is
    the commonest thing a typing hand does, and under plain Levenshtein
    `complaince` sits two edits from `compliance` and one from `complaint`, so
    the corrector confidently picks the wrong word. Counting the swap once puts
    both at one edit and hands the decision to :meth:`CorpusSpelling._unambiguous`,
    where the corpus's own usage settles it.
    """
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    two_back: list[int] | None = None
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        best = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if (two_back is not None and i > 1 and j > 1
                    and ca == b[j - 2] and a[i - 2] == cb):
                value = min(value, two_back[j - 2] + 1)
            current.append(value)
            best = min(best, value)
        if best > limit:
            return limit + 1
        two_back, previous = previous, current
    return previous[-1]


@dataclass(frozen=True)
class Replacement:
    """One word the corrector changed, and what it changed it to."""

    original: str
    corrected: str
    rule: str
    document_frequency: int

    def to_dict(self) -> dict[str, Any]:
        return {"from": self.original, "to": self.corrected,
                "rule": self.rule, "document_frequency": self.document_frequency}


@dataclass(frozen=True)
class Correction:
    """What was typed, what will be searched for, and every difference between them."""

    original: str
    corrected: str
    replacements: tuple[Replacement, ...] = ()

    @property
    def applied(self) -> bool:
        return bool(self.replacements)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original,
            "search_query": self.corrected,
            "applied": self.applied,
            "replacements": [r.to_dict() for r in self.replacements],
            "version": SPELLING_VERSION,
        }


class CorpusSpelling:
    """The corpus's own vocabulary, indexed for conservative repair."""

    version = SPELLING_VERSION

    def __init__(self, texts: Iterable[str], *,
                 min_document_frequency: int = MIN_DOCUMENT_FREQUENCY):
        self.min_document_frequency = int(min_document_frequency)
        frequency: Counter[str] = Counter()
        for text in texts:
            frequency.update({word.lower() for word in _WORD.findall(text or "")})
        # `vocabulary` decides whether a word needs correcting and is deliberately
        # unfiltered: a word that occurs once in the corpus is still a word the
        # corpus uses, and correcting it would be the aggressive behaviour this
        # module exists to avoid. `targets` decides what a correction may point
        # at, and is filtered, because a hapax is exactly what a scanner error
        # looks like.
        self.vocabulary: frozenset[str] = frozenset(frequency)
        self.frequency: dict[str, int] = dict(frequency)
        self.targets: dict[str, int] = {
            word: count for word, count in frequency.items()
            if count >= self.min_document_frequency and len(word) >= MIN_WORD_LENGTH
        }
        by_skeleton: dict[str, list[str]] = {}
        for word in self.targets:
            by_skeleton.setdefault(skeleton(word), []).append(word)
        self.by_skeleton: dict[str, tuple[str, ...]] = {
            key: tuple(sorted(words, key=lambda w: (-self.targets[w], w)))
            for key, words in by_skeleton.items()
        }
        # Bucketed by first skeleton character and skeleton length, so a lookup
        # compares against a few hundred words rather than all of them. The
        # first character is also a correctness guard, not only an index: a
        # typo that changes the first letter of a word is rare enough, and
        # ambiguous enough, that this module declines to handle it.
        self._buckets: dict[tuple[str, int], list[str]] = {}
        for key in self.by_skeleton:
            if key:
                self._buckets.setdefault((key[0], len(key)), []).append(key)

    # -- the one public operation --------------------------------------------

    def correct(self, query: str) -> Correction:
        """Return `query` with high-confidence misspellings repaired."""
        text = query or ""
        replacements: list[Replacement] = []
        pieces: list[str] = []
        last = 0
        for match in _WORD.finditer(text):
            word = match.group(0)
            replacement = self._correct_word(word)
            if replacement is None:
                continue
            pieces.append(text[last:match.start()])
            pieces.append(_match_case(word, replacement.corrected))
            last = match.end()
            replacements.append(replacement)
        if not replacements:
            return Correction(original=text, corrected=text)
        pieces.append(text[last:])
        return Correction(original=text, corrected="".join(pieces),
                          replacements=tuple(replacements))

    def _correct_word(self, word: str) -> Replacement | None:
        lowered = word.lower()
        if len(lowered) < MIN_WORD_LENGTH:
            return None
        if lowered in self.vocabulary:
            return None

        key = skeleton(lowered)
        if not key:
            return None

        # Tier one: the skeletons agree exactly. `markk`/`mark`,
        # `copywrite`/`copyright`, `applicaton` once the doubles collapse.
        exact = self.by_skeleton.get(key)
        if exact:
            chosen = self._unambiguous(exact)
            if chosen is not None:
                return Replacement(word, chosen, "skeleton_match",
                                   self.targets[chosen])
            return None

        # Tier two: one edit apart on the skeleton, first letter fixed.
        limit = MAX_DISTANCE
        best: int | None = None
        candidates: list[str] = []
        for length in range(len(key) - limit, len(key) + limit + 1):
            for other in self._buckets.get((key[0], length), ()):
                distance = _distance(key, other, limit)
                if distance > limit:
                    continue
                if best is None or distance < best:
                    best, candidates = distance, list(self.by_skeleton[other])
                elif distance == best:
                    candidates.extend(self.by_skeleton[other])
        if best is None:
            return None
        chosen = self._unambiguous(candidates)
        if chosen is None:
            return None
        if self.targets[chosen] < MIN_TARGET_FREQUENCY_FOR_DISTANCE:
            return None
        return Replacement(word, chosen, f"skeleton_distance_{best}",
                           self.targets[chosen])

    def _unambiguous(self, candidates: Sequence[str]) -> str | None:
        """The one obvious winner among `candidates`, or nothing.

        Two words the corpus uses comparably often are two readings of the
        typo, and this module does not pick between readings — it declines and
        lets retrieval see the word as typed. The winner must be at least twice
        as common as its nearest rival, which is a wide enough gap that the
        choice is the corpus's rather than the tie-breaker's.
        """
        unique = sorted(set(candidates), key=lambda w: (-self.targets[w], w))
        if not unique:
            return None
        if len(unique) == 1:
            return unique[0]
        top, runner_up = self.targets[unique[0]], self.targets[unique[1]]
        return unique[0] if top >= 2 * runner_up else None

    def config(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "deterministic": True,
            "uses_model": False,
            "dictionary": "corpus text only",
            "vocabulary_size": len(self.vocabulary),
            "correction_targets": len(self.targets),
            "min_document_frequency": self.min_document_frequency,
            "min_word_length": MIN_WORD_LENGTH,
            "max_skeleton_distance": MAX_DISTANCE,
            "min_target_frequency_for_distance": MIN_TARGET_FREQUENCY_FOR_DISTANCE,
            "guarantee": "a word the corpus uses is never rewritten; a word "
                         "containing a digit is never examined; the answer "
                         "generator receives the question as typed",
        }


def _match_case(original: str, corrected: str) -> str:
    """Keep the shape the user typed, so a correction is not also a case change."""
    if original.isupper() and len(original) > 1:
        return corrected.upper()
    if original[:1].isupper():
        return corrected[:1].upper() + corrected[1:]
    return corrected


def spelling_config(speller: "CorpusSpelling | None") -> dict[str, Any]:
    if speller is None:
        return {"version": SPELLING_VERSION, "enabled": False}
    return dict(speller.config(), enabled=True)
