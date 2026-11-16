"""Shared vocabulary for the structural parsers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: A page-attributed line. Every downstream unit and chunk can therefore name the
#: physical PDF pages it came from.
Line = tuple[int, str]

#: The full locator vocabulary. A chunk populates the keys its instrument
#: actually uses and leaves the rest null; nothing is ever guessed.
LOCATOR_KEYS = (
    "part",
    "chapter",
    "section",
    "subsection",
    "clause",
    "rule",
    "sub_rule",
    "regulation",
    "article",
    "paragraph",
    "subparagraph",
    "schedule",
    "form",
    "annexure",
    #: A numbered entry inside a Schedule, Form or Annexure. Schedules restart
    #: numbering at 1 and their entries are not provisions of the parent
    #: instrument, so they must never borrow `section`, `rule` or `regulation`:
    #: the Patents Act Schedule's entry 3 ("Section 1—In sub-section (1) omit
    #: ...") is an amendment instruction, not section 3.
    "item",
)


def empty_locator() -> dict[str, Any]:
    return {key: None for key in LOCATOR_KEYS}


@dataclass
class Unit:
    """One structural unit of an instrument — normally a whole provision."""

    unit_type: str
    number: str | None
    heading: str | None
    lines: list[Line]
    locator: dict[str, Any] = field(default_factory=empty_locator)
    amendment_notes: list[str] = field(default_factory=list)
    structural_path: str = ""

    @property
    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "\n".join(t for _, t in self.lines)).strip()

    @property
    def pages(self) -> list[int]:
        return sorted({p for p, _ in self.lines})

    def is_empty(self) -> bool:
        return not self.text.strip()


def parse_number(raw: str) -> tuple[int, str]:
    """Split a provision number such as ``11A`` into its sortable parts."""
    match = re.match(r"^(\d+)([A-Za-z]*)$", raw.strip())
    if not match:
        return (0, raw.strip())
    return int(match.group(1)), match.group(2).upper()


def accepts_next(previous: tuple[int, str] | None, candidate: tuple[int, str],
                 max_jump: int = 30) -> bool:
    """Is ``candidate`` a plausible next provision number after ``previous``?

    Statutes number their provisions monotonically. Requiring that guards against
    the commonest false positive in legal PDFs — an enumerated list inside a
    proviso ("1. Name of the Drug, 2. Batch number, ...") being mistaken for the
    start of sections 1 and 2.
    """
    if previous is None:
        return candidate[0] <= 3
    if candidate <= previous:
        return False
    return candidate[0] - previous[0] <= max_jump


def extract_heading(remainder: str) -> str | None:
    """Pull the marginal heading off the front of a provision.

    India Code sets provisions as ``3. What are not inventions.—The following
    are not inventions...``; the heading is what precedes the em dash.
    """
    match = re.match(r"^\s*(?P<h>[^\n]{2,140}?)\s*\.\s*[—–\-]{1,3}\s*\S", remainder)
    if not match:
        match = re.match(
            r"^\s*(?P<h>[^\n]{2,140}?)\s*\.\s+(?=[A-Z“\(])", remainder
        )
    if not match:
        # A heading occupying the whole line, with the text beginning on the
        # next: "1. Short title, extent and commencement." (Drugs and Cosmetics
        # Rules, and most pre-India-Code gazette typesetting).
        match = re.match(r"^\s*(?P<h>[^\n]{2,90}?)\s*\.\s*$", remainder)
    if not match:
        return None
    heading = re.sub(r"\s+", " ", match.group("h")).strip(" .—–-")
    if not heading or len(heading.split()) > 22:
        return None
    # A "heading" containing operative verbs is really the opening of the text.
    if re.search(r"(?i)\b(shall|means|includes|may be|is hereby)\b", heading):
        return None
    # A citation of another enactment is not a marginal heading. India Code
    # front matter lists amending Acts as "3. The Patents (Amendment) Act, 2002
    # (38 of 2002)." — numbered exactly like a section, and not one.
    if re.search(r"\(\s*\d{1,3}\s+of\s+(?:18|19|20)\d{2}\s*\)", heading):
        return None
    return heading
