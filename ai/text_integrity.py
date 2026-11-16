"""Whether a span of corpus text survived extraction intact — and nothing about repairing it.

Roughly one chunk in twenty of this corpus is damaged, and it is damaged before
any code in this repository sees it. Two distinct mechanisms, both at PDF
extraction:

*Dot-leader OCR.* The statutory Forms in the Patents Rules, 2003 are image scans
whose blank fill lines are rows of full stops. Rendered at 300 dpi a row of full
stops is a row of small disconnected blobs, and Tesseract maps them onto the
nearest low-x-height letters, so `..........................` comes back as
`eee eee eee ee ee` or `oo. cccccccc`. The affected pages are exactly the
low-confidence ones (mean OCR confidence 62-71 against a document mean of 82).

*Per-glyph positioning.* `2016DrugsandCosmeticsAct1940Rules1945.pdf` emits one
text-positioning operator per one or two glyphs — 1,604 of them for 3,481
characters on a single page — with the character-spacing parameter changing
between fragments. Any extractor has to infer word boundaries from horizontal
gaps, so `being` arrives as `be in g` and `a true copy of a letter` as
`A t ru e c o p y of a le t t e r`. This is not a library defect: poppler's
`pdftotext`, an unrelated engine, reproduces it character for character.

**This module detects. It never repairs.** That division is the whole point.
Repairing would mean deciding what the statute says on the basis of what it
looks like it might say, and the first time that is wrong it is wrong inside
quotation marks, attributed to an Act, in the one part of the interface whose
entire purpose is to be checkable. So a damaged span is shown as damaged, or not
shown, and the reader is told which. Nothing here rewrites a character.

What is *not* treated as damage matters as much as what is. Three signals were
tested against all 3,073 chunks and rejected:

* **Whitespace ratio.** Its top-ranked chunks are the bilingual Ayurveda and
  Unani dosage tables in the Drugs and Cosmetics Rules — correct extractions of
  a two-column layout.
* **Dictionary-word ratio.** Same failure, plus every schedule of chemical
  names.
* **Dot-leader runs.** 202 chunks (6.6%) contain `.........................`,
  and in almost all of them it is the literal blank-fill line of a statutory
  form. Flagging those would condemn a fifteenth of the corpus for being
  faithful to the page.

The three signals kept are the ones that cannot occur in correctly extracted
English legal prose at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

INTEGRITY_VERSION = "0.1.0"

CLEAN = "CLEAN"
DAMAGED = "EXTRACTION_DAMAGED"

#: A whole token that is one letter repeated: `ee`, `eee`, `ccccccc`. English
#: has no such word; OCR'd dot leaders are full of them.
_REPEATED_LETTER = re.compile(r"^([A-Za-z])\1+$")

#: Short tokens that are genuinely words, so that `of a letter` is not read as
#: `of` plus an orphaned fragment. Roman numerals and single letters used as
#: clause labels are here too, because `(a)`, `(i)` and `(b)` are everywhere in
#: this corpus — though those arrive parenthesised and are stripped out before
#: this set is consulted.
_REAL_SHORT_WORDS = frozenset("""
a an as at be by do go he he if in is it me my no of on or so to up us we
am are and any but can far for had has her him his how its may nor not now
off one our out own per put see she the too two use was way who why yes yet
i ii iii iv v vi vii viii ix x xi xii xiii xiv xv
""".split())

#: Characters stripped from a token's edges before it is classified, so that
#: `(a)`, `copy,` and `letter.` are judged on their letters.
_EDGE = " \t\r\n.,;:()[]{}\"'/\\-–—‐‑·•*"

#: Thresholds. Each is a count within the span being judged, and each was set
#: from the measured corpus census rather than chosen in advance:
#:
#: * Three repeated-letter tokens. One or two occur in correctly extracted text
#:   (`Ltd`, initials, `AA` in a schedule code); three in one span does not.
#: * A run of four consecutive single-letter tokens. Clause labels never run
#:   four deep unparenthesised, and `c o p y` always is damage.
#: * Three orphaned word fragments. One can be a genuine hyphenation artefact
#:   at a line break; three in one span is the per-glyph positioning defect.
MIN_REPEATED_LETTER_TOKENS = 3
MIN_LETTERSPACED_RUN = 4
MIN_ORPHAN_SPLITS = 3

#: Below this many tokens a span is too short for the counts above to mean
#: anything, and the rate test takes over: a five-token span that is three
#: repeated-letter tokens is damage, and the absolute threshold would miss it.
SHORT_SPAN_TOKENS = 25
SHORT_SPAN_DAMAGE_RATE = 0.25


@dataclass(frozen=True)
class TextIntegrity:
    """What extraction did to a span, stated in counts rather than adjectives."""

    status: str
    signals: tuple[str, ...]
    tokens: int
    repeated_letter_tokens: int
    letterspaced_runs: int
    orphan_splits: int

    @property
    def damaged(self) -> bool:
        return self.status == DAMAGED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "signals": list(self.signals),
            "tokens": self.tokens,
            "repeated_letter_tokens": self.repeated_letter_tokens,
            "letterspaced_runs": self.letterspaced_runs,
            "orphan_splits": self.orphan_splits,
            "repairable": False,
            "version": INTEGRITY_VERSION,
        }


def _strip(token: str) -> str:
    return token.strip(_EDGE)


def _is_repeated_letter_token(token: str) -> bool:
    """Is `token` a bare run of one letter — `ee`, `ccc`, `wwwww`?

    Judged on the raw token for the same reason as :func:`_is_orphan`: `(ii)`
    and `(iii)` are Roman clause labels, and stripping their brackets first
    turns every sub-clause in the corpus into OCR debris. Bare `ii` and `iii`
    are excluded too, via :data:`_REAL_SHORT_WORDS`, because statutes number
    clauses that way unbracketed as well. Bare `ss` is not excluded, and is not
    a numeral: it is the tail of `addre ss`.
    """
    if not token.isalpha() or len(token) < 2:
        return False
    if token.lower() in _REAL_SHORT_WORDS:
        return False
    return bool(_REPEATED_LETTER.match(token))


def _is_orphan(previous: str, token: str) -> bool:
    """Is `token` the tail of a word that extraction split in two?

    Judged on the tokens *as written*, before delimiters are stripped, because
    the delimiters are the evidence. `(f)` and `(g)` are clause labels and this
    corpus is built out of them; strip the brackets first and every lettered
    sub-clause in the Geographical Indications Rules reads as a broken word,
    which is how a first attempt at this function condemned 22% of the corpus
    instead of 5%.

    So the fragment must be bare: alphabetic with nothing around it, lower-case
    (`GI`, `TM` and `AA` are abbreviations, not debris), not a real short word,
    and following a bare alphabetic word long enough to have had a tail taken
    off it. `manufacture r` and `licensin g` qualify. `India, (b)` does not,
    twice over.
    """
    if not token.isalpha() or not token.islower() or len(token) > 2:
        return False
    if token in _REAL_SHORT_WORDS:
        return False
    return previous.isalpha() and len(previous) >= 4


def assess(text: str) -> TextIntegrity:
    """Judge one span of extracted text. Pure, deterministic, allocation-light."""
    raw = (text or "").split()
    words = [token for token in (_strip(token) for token in raw) if token]
    total = len(words)

    repeated = 0
    orphans = 0
    longest_run = 0
    run = 0
    previous_raw = ""
    for token in raw:
        core = _strip(token)
        if not core:
            previous_raw = token
            continue

        if _is_repeated_letter_token(token):
            repeated += 1

        if core.isalpha() and len(core) == 1:
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 0

        if _is_orphan(previous_raw, token):
            orphans += 1
        previous_raw = token

    signals: list[str] = []
    if total and total < SHORT_SPAN_TOKENS:
        # Short spans are judged by rate, because three of anything is a large
        # fraction of a twenty-token span and no fraction at all of a thousand.
        if repeated / total >= SHORT_SPAN_DAMAGE_RATE:
            signals.append("repeated_letter_tokens")
        if longest_run >= MIN_LETTERSPACED_RUN:
            signals.append("letterspaced_run")
        if orphans / total >= SHORT_SPAN_DAMAGE_RATE:
            signals.append("orphan_splits")
    else:
        if repeated >= MIN_REPEATED_LETTER_TOKENS:
            signals.append("repeated_letter_tokens")
        if longest_run >= MIN_LETTERSPACED_RUN:
            signals.append("letterspaced_run")
        if orphans >= MIN_ORPHAN_SPLITS:
            signals.append("orphan_splits")

    return TextIntegrity(
        status=DAMAGED if signals else CLEAN,
        signals=tuple(signals),
        tokens=total,
        repeated_letter_tokens=repeated,
        letterspaced_runs=longest_run,
        orphan_splits=orphans,
    )


def integrity_config() -> dict[str, Any]:
    return {
        "version": INTEGRITY_VERSION,
        "deterministic": True,
        "repairs_text": False,
        "signals": ("repeated_letter_tokens", "letterspaced_run", "orphan_splits"),
        "rejected_signals": ("whitespace_ratio", "dictionary_word_ratio",
                             "dot_leader_runs"),
        "thresholds": {
            "min_repeated_letter_tokens": MIN_REPEATED_LETTER_TOKENS,
            "min_letterspaced_run": MIN_LETTERSPACED_RUN,
            "min_orphan_splits": MIN_ORPHAN_SPLITS,
            "short_span_tokens": SHORT_SPAN_TOKENS,
            "short_span_damage_rate": SHORT_SPAN_DAMAGE_RATE,
        },
        "guarantee": "detection only; no character of corpus text is ever "
                     "rewritten, and a damaged span is reported as damaged "
                     "rather than cleaned, shortened or replaced",
    }
