"""Parser for international instruments: TRIPS, the CBD, the Nagoya Protocol.

These use Article → paragraph → subparagraph, not Chapter → Section → sub-section,
and the Phase 1 audit found no Section, Rule or Schedule structure in any of them.
Running one parser over both grammars would mis-chunk all three.

The CBD and the Nagoya Protocol set their headings as a centred block, which the
text layer emits as three separate lines — ``Article`` / ``15`` / ``ACCESS TO
GENETIC RESOURCES`` — so a bare ``Article`` line is joined to the number below it.
"""
from __future__ import annotations

import re
from typing import Any

from ingest.parsers.base import Line, Unit, accepts_next, empty_locator, parse_number

ARTICLE_INLINE_RE = re.compile(r"^\s*Article\s+(\d{1,3}(?:\s?(?:bis|ter|quater))?)\s*$", re.I)
ARTICLE_BARE_RE = re.compile(r"^\s*Article\s*$", re.I)
#: The centred number under an ``Article`` heading. Digits may arrive separated
#: — the Nagoya Protocol's Article 11 reaches the text layer as ``1 1`` — because
#: the heading is letter-spaced in the original typesetting. Joining them reads
#: the printed number; it does not infer one.
NUMBER_ONLY_RE = re.compile(r"^\s*(\d(?:\s?\d){0,2}(?:\s?(?:bis|ter|quater))?)\s*$", re.I)
PART_RE = re.compile(r"^\s*PART\s+([IVXLCDM]+)\s*$", re.I)
SECTION_RE = re.compile(r"^\s*SECTION\s+(\d+)\s*:?\s*(.*)$", re.I)
ANNEX_RE = re.compile(r"^\s*ANNEX(?:URE)?\s*([IVXLCDM]+|\d+|[A-Z])?\s*$", re.I)
APPENDIX_RE = re.compile(r"^\s*APPENDIX\s*([IVXLCDM]+|\d+|[A-Z])?\s*$", re.I)

PARAGRAPH_RE = re.compile(r"^\s*(\d{1,2})\s*\.\s+(?=\S)")
SUBPARAGRAPH_RE = re.compile(r"^\s*\(\s*([a-z]{1,3})\s*\)")


def _is_title(text: str) -> bool:
    """Is this line the Article's rubric rather than its first paragraph?

    The CBD sets its rubrics in small caps, which the text layer emits as
    ``aCCess to genetiC resourCes`` — a case test alone would reject them — so a
    short line that neither ends a sentence nor opens a numbered paragraph is
    accepted as the title.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3 or len(text) > 120:
        return False
    if sum(1 for c in letters if c.isupper()) / len(letters) >= 0.5:
        return True
    return (
        len(text.strip()) <= 80
        and not text.strip().endswith((".", ";", ",", ":"))
        and not PARAGRAPH_RE.match(text)
        and not SUBPARAGRAPH_RE.match(text)
    )


def parse(lines: list[Line], provision_label: str = "Article") -> list[Unit]:
    units: list[Unit] = []
    state: dict[str, Any] = {"part": None, "section": None, "annexure": None}
    current = Unit(unit_type="preamble", number=None, heading=None, lines=[],
                   locator=empty_locator())
    consumed: set[int] = set()

    def close() -> None:
        nonlocal current
        if not current.is_empty():
            units.append(current)

    def open_unit(unit_type: str, number: str | None, heading: str | None) -> Unit:
        locator = empty_locator()
        locator["part"] = state["part"]
        locator["section"] = state["section"]
        locator["annexure"] = state["annexure"]
        if unit_type == "article" and number is not None:
            locator["article"] = number
        return Unit(unit_type=unit_type, number=number, heading=heading, lines=[],
                    locator=locator)

    index = 0
    while index < len(lines):
        if index in consumed:
            index += 1
            continue
        page, text = lines[index]
        stripped = text.strip()
        if not stripped:
            current.lines.append((page, text))
            index += 1
            continue

        article_number = None
        article_heading = False
        skip_to = index + 1
        inline = ARTICLE_INLINE_RE.match(stripped)
        if inline:
            article_heading = True
            article_number = re.sub(r"\s+", "", inline.group(1))
        elif ARTICLE_BARE_RE.match(stripped):
            article_heading = True
            for look in range(index + 1, min(index + 4, len(lines))):
                candidate = lines[look][1].strip()
                if not candidate:
                    continue
                number_match = NUMBER_ONLY_RE.match(candidate)
                if number_match:
                    article_number = re.sub(r"\s+", "", number_match.group(1))
                    consumed.update(range(index + 1, look + 1))
                    skip_to = look + 1
                break

        # A heading with no recoverable number still opens its own Article. The
        # alternative — folding the text into whichever unit preceded it — files
        # one Article's words under another Article's citation, which is the
        # worst outcome available. Opened without a number, the unit reaches
        # Phase 3 as `locator_unavailable` and is therefore not citable.
        if article_heading:
            close()
            title = None
            for look in range(skip_to, min(skip_to + 3, len(lines))):
                candidate = lines[look][1].strip()
                if not candidate:
                    continue
                if _is_title(candidate):
                    title = re.sub(r"\s+", " ", candidate)
                break
            current = open_unit("article", article_number, title)
            current.lines.append((page, text))
            index = skip_to
            continue

        part = PART_RE.match(stripped)
        section = SECTION_RE.match(stripped)
        annexure = ANNEX_RE.match(stripped)
        appendix = APPENDIX_RE.match(stripped)

        if part or section or annexure or appendix:
            close()
            if part:
                state.update({"part": part.group(1).upper(), "section": None})
                unit_type, number = "heading", None
            elif section:
                state["section"] = section.group(1)
                unit_type, number = "heading", None
            elif annexure:
                state["annexure"] = (annexure.group(1) or "ANNEX").upper()
                unit_type, number = "annexure", state["annexure"]
            else:
                state["annexure"] = f"APPENDIX {appendix.group(1) or ''}".strip()
                unit_type, number = "annexure", state["annexure"]
            current = open_unit(unit_type, number, re.sub(r"\s+", " ", stripped))
            current.lines.append((page, text))
            index += 1
            continue

        current.lines.append((page, text))
        index += 1

    close()
    for unit in units:
        unit.structural_path = build_path(unit)
    return units


def build_path(unit: Unit) -> str:
    parts = []
    loc = unit.locator
    if loc.get("part"):
        parts.append(f"Part {loc['part']}")
    if loc.get("section"):
        parts.append(f"Section {loc['section']}")
    if loc.get("annexure"):
        parts.append(str(loc["annexure"]))
    if loc.get("article"):
        parts.append(f"Article {loc['article']}")
    return " > ".join(parts)


def subdivisions(lines: list[Line], provision_key: str = "article") -> list[tuple[int, str, str]]:
    """Numbered-paragraph boundaries inside an Article."""
    out: list[tuple[int, str, str]] = []
    previous: tuple[int, str] | None = None
    for index, (_, text) in enumerate(lines):
        match = PARAGRAPH_RE.match(text)
        if not match:
            continue
        candidate = parse_number(match.group(1))
        if accepts_next(previous, candidate, max_jump=10):
            previous = candidate
            out.append((index, "paragraph", match.group(1)))
    return out
