"""Phase F: the short exact passage a source card shows by default.

A chunk in this corpus runs to two or three printed pages. Putting the whole of
one under an answer is technically honest and practically useless: the reader is
handed the haystack and told the needle is in it. What a citation has to show,
before anything else, is *the sentence the answer rests on* — and then let the
reader open the rest.

Three properties, and the module is built around them rather than around a
length target:

**The excerpt is a substring of the retrieved chunk.** Never a paraphrase, never
a regenerated quotation, never a model's recollection of a passage. It is sliced
out of the same `text` the answer provider was shown, and `select_excerpt`
returns the character offsets so a caller can assert that or highlight it.
Precisely: every excerpt satisfies
``excerpt.text == display_text(chunk)[excerpt.start:excerpt.end].strip()``,
where `display_text` collapses runs of spaces and tabs and touches nothing
else — a caller checking the guarantee must normalise the same way, because
this corpus is full of OCR'd words like "Regis ter" whose internal spacing is
not stable. If
this module cannot find a sentence it is confident about, it returns a *window*
around the best match and says so in `strategy` — which is the Phase F fallback,
chosen deliberately over trimming a passage until it looks tidy.

**The legal wording inside it is untouched.** No ellipsis inside a span, no
case-folding, no whitespace rewriting beyond the newline collapsing the chunker
already applied for display. An excerpt that has been cleaned up is a quotation
of something the statute does not say.

**It never replaces the full passage.** `Excerpt.text` is what a card shows
first; the chunk stays on the response beside it for "View full passage". A
reader who wants to check the excerpt in context must always be able to.

Selection is deterministic — no model, no network, no state. It scores each
sentence on how much of the *question* and of the *answer's own wording* it
accounts for, which is the pair of signals that distinguishes the operative
sentence from the definitional throat-clearing around it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ai.answering.grounding import CITATION_MARKER, split_sentences
from ai.answering.policy import content_terms
from ai.text_integrity import assess as assess_integrity

EXCERPT_VERSION = "0.1.0"

#: The longest excerpt a card shows before "View full passage". Two or three
#: sentences of statutory drafting; past that a reader is skim-reading again,
#: which is the problem this module exists to fix. It bounds how many sentences
#: are joined, not whether a single long one is shown whole — see `select_excerpt`.
MAX_EXCERPT_CHARS = 460

#: The shortest span worth showing on its own. A sub-clause of six words read
#: without its stem is not evidence, so below this the window widens instead.
MIN_EXCERPT_CHARS = 80

#: Half-width of the fallback window, in characters, measured around the best
#: matching position when sentence splitting produced nothing usable.
WINDOW_RADIUS = 220

#: Past this multiple of the cap, a "sentence" is not a sentence. Statutory
#: drafting does produce single sentences of 600-900 characters — s. 3 of the
#: Patents Act is one — and those are shown whole, because cutting one would
#: strip a proviso off its rule. A span several times longer than that is an
#: OCR'd page or a table the splitter could find no full stop in, and returning
#: it whole would hand the reader the two pages Phase F exists to stop showing.
#: Those go to the window, which says what it is.
SENTENCE_LENGTH_LIMIT = 3.0

_WHITESPACE = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class Excerpt:
    """One short verbatim span of a chunk, with its place in that chunk."""

    text: str
    start: int
    end: int
    strategy: str
    sentence_count: int
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "strategy": self.strategy,
            "sentence_count": self.sentence_count,
            "truncated": self.truncated,
            "version": EXCERPT_VERSION,
        }


def display_text(text: str) -> str:
    """The chunk as a card renders it: runs of spaces collapsed, lines kept.

    Offsets are reported against *this* string, so a caller that highlights the
    excerpt highlights it in the text it is actually showing. Only horizontal
    whitespace is touched, and it is collapsed rather than removed, so no word
    boundary moves and every character of the legal wording survives.
    """
    return _WHITESPACE.sub(" ", text or "")


#: Kept as a private alias because this module's own call sites and its tests
#: were written against it before the name became part of the contract.
_display_text = display_text


def _answer_terms(answer_text: str | None, marker: int | None) -> frozenset[str]:
    """Content terms of the answer sentences that cite this passage.

    Restricted to the sentences carrying this marker, because an answer over
    five passages says different things about each, and scoring a passage
    against all of it would pull every excerpt towards whatever the answer
    talks about most.
    """
    if not answer_text:
        return frozenset()
    if marker is None:
        return content_terms(answer_text)
    wanted = [sentence for sentence in split_sentences(answer_text)
              if marker in {int(m.group(1)) for m in CITATION_MARKER.finditer(sentence)}]
    return content_terms(" ".join(wanted)) if wanted else frozenset()


def _score(sentence: str, query_terms: frozenset[str],
           answer_terms: frozenset[str]) -> float:
    """How much of the question and of the answer this sentence accounts for.

    The question is weighted above the answer deliberately. The answer was
    written *from* this passage, so scoring on it alone rewards whichever
    sentence the provider happened to echo; the question is the independent
    signal, and it is the reader's actual interest.
    """
    terms = content_terms(sentence)
    if not terms:
        return 0.0
    query_hit = len(terms & query_terms) / len(query_terms) if query_terms else 0.0
    answer_hit = len(terms & answer_terms) / len(answer_terms) if answer_terms else 0.0
    # A one-clause fragment can score 1.0 on a two-word query; the length term
    # breaks those ties towards the sentence that actually states the rule.
    substance = min(len(sentence) / float(MAX_EXCERPT_CHARS), 1.0)
    return 0.6 * query_hit + 0.3 * answer_hit + 0.1 * substance


def _anchor(display: str, query_terms: frozenset[str]) -> int:
    """Where in `display` the query is most concentrated.

    The first version of this took the *earliest* occurrence of any query term,
    which is only the right answer when the chunk mentions the subject once. On
    a five-thousand-character schedule where a common word like "fee" appears in
    the header, the earliest hit opens the window four thousand characters away
    from the passage the reader wants, and the card shows the top of the page.

    So every occurrence of every term is collected, and the anchor is the one
    whose surrounding window covers the most *distinct* query terms — hit count
    breaks ties, then position, so the result is deterministic. Density is the
    signal that matters: a place where four of the query's words appear together
    is the passage; a place where one common word appears alone is not.
    """
    if not query_terms:
        return 0
    lowered = display.lower()
    occurrences: list[tuple[int, str]] = []
    for term in sorted(query_terms):
        position = lowered.find(term)
        while position >= 0:
            occurrences.append((position, term))
            position = lowered.find(term, position + 1)
    if not occurrences:
        return 0
    occurrences.sort()
    best_key = (-1, -1, 0)
    for position, _ in occurrences:
        low, high = position - WINDOW_RADIUS, position + WINDOW_RADIUS
        covered = {term for where, term in occurrences if low <= where <= high}
        hits = sum(1 for where, _ in occurrences if low <= where <= high)
        key = (len(covered), hits, -position)
        if key > best_key:
            best_key = key
            anchor = position
    return anchor


def _window(display: str, query_terms: frozenset[str]) -> Excerpt:
    """The Phase F fallback: a span around the best match, on word boundaries.

    Used when sentence splitting yields nothing usable — an OCR'd page with no
    full stops, a schedule that is really a table. It is explicitly a *window*
    and says so, rather than being presented as a sentence it is not.
    """
    anchor = _anchor(display, query_terms)
    start = max(0, anchor - WINDOW_RADIUS)
    end = min(len(display), start + 2 * WINDOW_RADIUS)
    # Never cut a word in half: walk out to the nearest whitespace.
    if start > 0:
        space = display.find(" ", start)
        start = space + 1 if 0 <= space < start + 40 else start
    if end < len(display):
        space = display.rfind(" ", end - 40, end)
        end = space if space > start else end
    return Excerpt(text=display[start:end].strip(), start=start, end=end,
                   strategy="window", sentence_count=0,
                   truncated=(start > 0 or end < len(display)))


def select_excerpt(text: str, query: str, *, answer_text: str | None = None,
                   marker: int | None = None,
                   max_chars: int = MAX_EXCERPT_CHARS) -> Excerpt | None:
    """The span of `text` a source card should show first, or None if empty.

    The returned `text` is always `display_text(text)[start:end]` — assert it if
    you like; `tests/test_excerpt.py` does.
    """
    display = display_text(text)
    if not display.strip():
        return None
    if len(display) <= max_chars:
        return Excerpt(text=display.strip(), start=0, end=len(display),
                       strategy="whole_chunk", sentence_count=len(split_sentences(display)),
                       truncated=False)

    query_terms = content_terms(query)
    answer_terms = _answer_terms(answer_text, marker)
    sentences = split_sentences(display)
    if not sentences:
        return _window(display, query_terms)

    # Offsets by scanning forward, so a repeated sentence maps to its own
    # position rather than to the first identical one.
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for sentence in sentences:
        position = display.find(sentence, cursor)
        if position < 0:
            position = display.find(sentence)
            if position < 0:
                continue
        spans.append((position, position + len(sentence), sentence))
        cursor = position + len(sentence)
    if not spans:
        return _window(display, query_terms)

    scores = [_score(sentence, query_terms, answer_terms) for _, _, sentence in spans]
    best = max(range(len(spans)), key=lambda index: (scores[index], -index))
    start, end, _ = spans[best]

    # A single statutory sentence can run past `max_chars` on its own — s. 3 of
    # the Patents Act is one sentence with sixteen lettered clauses. It is
    # returned whole. Cutting it at a character count would end the quotation
    # mid-clause, which is exactly the "never alter the legal wording" rule this
    # module exists to keep: a reader would see a requirement without its
    # proviso and have no way to tell. The cap governs how many sentences are
    # joined, not whether one is shown intact.
    if end - start >= max_chars:
        if end - start > SENTENCE_LENGTH_LIMIT * max_chars:
            return _window(display, query_terms)
        return Excerpt(text=display[start:end].strip(), start=start, end=end,
                       strategy="sentence", sentence_count=1,
                       truncated=(start > 0 or end < len(display)))

    # Grow into the neighbour that adds the most, while the span still fits. A
    # statutory sentence often carries its condition in the one before it.
    count = 1
    while count < 3:
        options = []
        if best - count >= 0:
            previous = spans[best - count]
            if end - previous[0] <= max_chars:
                options.append((scores[best - count], previous[0], end))
        after = best + count
        if after < len(spans) and spans[after][1] - start <= max_chars:
            options.append((scores[after], start, spans[after][1]))
        if not options:
            break
        _, start, end = max(options)
        count += 1

    if end - start < MIN_EXCERPT_CHARS:
        end = min(len(display), start + max_chars)
        space = display.rfind(" ", end - 40, end)
        end = space if space > start else end
    span = display[start:end]
    return Excerpt(text=span.strip(), start=start, end=end, strategy="sentence",
                   sentence_count=count, truncated=(start > 0 or end < len(display)))


def excerpt_for_evidence(item: Any, query: str, *,
                         answer_text: str | None = None) -> dict[str, Any] | None:
    """`select_excerpt` for one `Evidence`, keyed on that passage's marker.

    The excerpt is also assessed for extraction damage, and the verdict travels
    with it. Selection itself is deliberately *not* steered away from damaged
    text: the excerpt's job is to show the passage the answer rests on, and
    quietly substituting a cleaner sentence from elsewhere in the chunk would
    put a tidy quotation under a claim it does not support. A damaged passage is
    shown as damaged, and the card says so.
    """
    excerpt = select_excerpt(getattr(item, "text", "") or "", query,
                             answer_text=answer_text,
                             marker=getattr(item, "marker", None))
    if excerpt is None:
        return None
    return dict(excerpt.to_dict(), integrity=assess_integrity(excerpt.text).to_dict())


def excerpt_config() -> dict[str, Any]:
    return {
        "version": EXCERPT_VERSION,
        "max_excerpt_chars": MAX_EXCERPT_CHARS,
        "min_excerpt_chars": MIN_EXCERPT_CHARS,
        "window_radius": WINDOW_RADIUS,
        "sentence_length_limit": SENTENCE_LENGTH_LIMIT,
        "guarantee": "every excerpt is a verbatim substring of the retrieved "
                     "chunk; offsets index the chunk as displayed; extraction "
                     "damage is reported, never repaired",
        "strategies": ("whole_chunk", "sentence", "window"),
    }
