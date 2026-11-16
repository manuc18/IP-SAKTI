"""Deterministic, human-readable citation locators.

The locator — not a PDF page number — is the primary citation. Page numbers are
kept alongside as supporting metadata, because they identify a rendering of a
document rather than a place in the law.

Three grammars are in play, and each instrument is cited the way its own
community cites it:

* ``indian``  Patents Act, 1970 — Section 3(p) / Patents Rules, 2003 — Rule 13(2)
* ``dot``     TRIPS — Article 27.1 / Nagoya Protocol — Article 5.2
* ``paren``   CBD — Article 15(7)

A citation must also *distinguish*. Two provisions that render the same string
are two provisions a reader cannot tell apart, however well the chunk ids
separate them — see :func:`disambiguate_citations`.
"""
from __future__ import annotations

from typing import Any

LABEL_FOR_KEY = {
    "section": "Section",
    "rule": "Rule",
    "regulation": "Regulation",
    "article": "Article",
    "paragraph": "Paragraph",
}

#: Structural fields that may be promoted into a citation to separate two
#: provisions that would otherwise render identically, in the order they are
#: tried. Both are containers a reader already navigates by, so promoting one
#: adds a coordinate rather than inventing a new grammar: "Part VI, Rule 65" is
#: how the instrument itself is organised. The first field whose values actually
#: differ across the colliding group wins; nothing further is tried, because a
#: citation carrying two redundant coordinates is harder to read, not clearer.
DISAMBIGUATION_FIELDS = (("part", "Part"), ("chapter", "Chapter"))


def build_citation(
    short_title: str,
    locator: dict[str, Any],
    *,
    citation_style: str = "indian",
    unit_type: str = "provision",
    pages: list[int] | None = None,
    structural_prefix: str | None = None,
) -> tuple[str, bool]:
    """Return ``(citation, locator_unavailable)``.

    `structural_prefix` — "Part VI", "Chapter IV" — is prepended to the locator
    when this provision would otherwise be indistinguishable from another in the
    same instrument. It is supplied by :func:`disambiguate_citations`, which is
    the only place that can know a collision exists, and is never guessed here:
    a citation that carries a Part because the chunker happened to record one
    would be noisier than the community's own convention for no gain.
    """
    if unit_type in ("schedule", "form", "annexure"):
        label = _structural_label(locator, unit_type)
        if label:
            return f"{short_title} — {_head(structural_prefix)}{label}", False

    # A numbered entry inside a Schedule, Form or Annexure is cited through its
    # container — "Patents Act, 1970 — Schedule, item 3" — never as a section of
    # the parent instrument. The container is what makes the entry findable; an
    # entry number on its own identifies nothing.
    if unit_type == "schedule_item" and locator.get("item"):
        label = _structural_label(locator, unit_type)
        if label:
            return (f"{short_title} — {_head(structural_prefix)}{label}, "
                    f"item {locator['item']}"), False
        return f"{short_title} — unlocated text", True

    for key in ("section", "rule", "regulation", "article", "paragraph"):
        number = locator.get(key)
        if not number:
            continue
        tail = _tail(locator, key, citation_style)
        prefix = _structural_label(locator, unit_type)
        head = f"{prefix}, " if prefix and key != "paragraph" else ""
        return (f"{short_title} — {_head(structural_prefix)}{head}"
                f"{LABEL_FOR_KEY[key]} {number}{tail}"), False

    label = _structural_label(locator, unit_type)
    if label:
        return f"{short_title} — {_head(structural_prefix)}{label}", False

    # A Chapter or Part heading is itself citable, even with no provision under it.
    for key, word in (("chapter", "Chapter"), ("part", "Part")):
        if locator.get(key):
            return f"{short_title} — {word} {locator[key]}", False

    page_note = f" (physical p. {pages[0]})" if pages else ""
    return f"{short_title} — unlocated text{page_note}", True


def disambiguate_citations(chunks: list[dict[str, Any]], *, short_title: str,
                           citation_style: str = "indian") -> int:
    """Re-render citations that two different provisions would share.

    Returns the number of chunks whose citation changed. Operates on a whole
    document's chunks in place, because a collision is not visible from one
    chunk — which is why this cannot live inside :func:`build_citation`.

    **A collision is not the same as a repeated citation.** A long provision is
    split across several chunks, and all of them are genuinely Rule 65; giving
    them different citations would be a lie about the corpus, not a fix. The
    test is therefore not "does this string repeat" but "do the chunks sharing
    it sit in *different structural containers*" — and a split provision has one
    locator, so no field separates it and nothing is touched.

    Worked example, and the reason this exists. The Drugs and Cosmetics Rules,
    1945 contain Part VI rule 65 (conditions of licence, pp. 71-72) and Part II
    rule 65 (a schedule entry, p. 533). Both rendered as
    ``Drugs and Cosmetics Rules, 1945 — Rule 65``. The chunk ids always
    distinguished them and the citation objects always resolved; the string a
    reader sees did not. After this pass they read ``Part VI, Rule 65`` and
    ``Part II, Rule 65``.

    A chunk in the colliding group that has no value for the separating field
    keeps its original citation. There is nothing truthful to promote for it,
    and inventing a container to make the set look symmetrical would be the
    fabrication this whole layer exists to prevent.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        groups.setdefault(chunk["citation"], []).append(chunk)

    changed = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        for field, label in DISAMBIGUATION_FIELDS:
            # Two distinct values separate the group, and "recorded" versus
            # "not recorded" counts as two: promoting the Part onto the chunk
            # that has one still leaves a reader able to tell the pair apart,
            # which is the whole objective. No such case exists in the corpus
            # today; requiring two *present* values would silently do nothing if
            # one ever appeared.
            values = {_normalise(chunk["locator"].get(field)) for chunk in group}
            if len(values) < 2:
                continue
            for chunk in group:
                value = chunk["locator"].get(field)
                if not value:
                    continue
                citation, unavailable = build_citation(
                    short_title, chunk["locator"],
                    citation_style=citation_style,
                    unit_type=chunk["unit_type"],
                    pages=chunk["physical_pages"],
                    structural_prefix=f"{label} {value}",
                )
                if citation != chunk["citation"]:
                    chunk["citation"] = citation
                    chunk["locator_unavailable"] = unavailable
                    changed += 1
            break
    return changed


def _normalise(value: Any) -> str | None:
    text = str(value).strip().upper() if value not in (None, "") else None
    return text or None


def _head(structural_prefix: str | None) -> str:
    return f"{structural_prefix}, " if structural_prefix else ""


def _structural_label(locator: dict[str, Any], unit_type: str) -> str:
    if locator.get("form"):
        return f"Form {locator['form']}"
    if locator.get("schedule"):
        schedule = str(locator["schedule"]).title()
        return "Schedule" if schedule.upper() == "SCHEDULE" else f"{schedule} Schedule"
    if locator.get("annexure"):
        annexure = str(locator["annexure"])
        return annexure.title() if annexure.upper().startswith(("ANNEX", "APPENDIX")) else f"Annexure {annexure}"
    return ""


def _tail(locator: dict[str, Any], key: str, citation_style: str) -> str:
    if key == "article":
        paragraph = locator.get("paragraph")
        subparagraph = locator.get("subparagraph")
        tail = ""
        if paragraph:
            tail += f".{paragraph}" if citation_style == "dot" else f"({paragraph})"
        if subparagraph:
            tail += f"({subparagraph})"
        return tail
    tail = ""
    for sub_key in ("subsection", "sub_rule"):
        if locator.get(sub_key):
            tail += f"({locator[sub_key]})"
            break
    if locator.get("clause"):
        tail += f"({locator['clause']})"
    return tail
