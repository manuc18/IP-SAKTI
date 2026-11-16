"""Parser for Indian statutory instruments: Acts, Rules, Regulations, Policies.

The hierarchy this recognises is the one Indian drafting actually uses —
Part / Chapter → Section, Rule or Regulation → sub-section → clause, with
Schedules, Forms and Annexures as free-standing units that restart their own
numbering. Treaties are a different grammar and are parsed elsewhere.
"""
from __future__ import annotations

import re
from typing import Any

from ingest.parsers.base import (
    Line,
    Unit,
    accepts_next,
    empty_locator,
    extract_heading,
    parse_number,
)

CHAPTER_RE = re.compile(r"^\s*CHAPTER\s+([IVXLCDM]+|\d+[A-Z]?)\s*$", re.I)
PART_RE = re.compile(r"^\s*PART\s+([IVXLCDM]+|\d+[A-Z]?)\s*$", re.I)
ORDINALS = (
    "FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|"
    "ELEVENTH|TWELFTH|THIRTEENTH|FOURTEENTH"
)
SCHEDULE_RE = re.compile(
    rf"^\s*(?:THE\s+)?(?:({ORDINALS})\s+)?SCHEDULE\s*(?:[-–—]?\s*([A-Z]|[IVXLCDM]+))?\s*$",
    re.I,
)
#: A Form heading stands alone on its line. Anchoring to end-of-line matters:
#: operative text routinely opens a line with "Form 4 for extension of time...",
#: and treating that as a heading swallows every rule that follows it.
FORM_RE = re.compile(r"^\s*FORM\s+(?:No\.?\s*)?(\d+[A-Z]*)\s*[-–—:.]?\s*$", re.I)
ANNEX_RE = re.compile(r"^\s*ANNEX(?:URE)?\s*[-–—]?\s*([IVXLCDM]+|\d+|[A-Z])?\s*$", re.I)

#: A provision opener. The optional ``N[`` prefix is an India Code footnote
#: marker that the text layer flattens onto the line ("2[44. Qualifications...").
#: Up to four trailing letters: Indian drafting inserts sections as 33E, 33EE,
#: 33EEA, 33EEB when it amends between existing ones.
PROVISION_RE = re.compile(r"^\s*(?:\d{1,2}\s*\[\s*)?(\d{1,3}[A-Z]{0,4})\s*\.\s*(?=\S)")

#: How far a provision number may leap before it needs corroboration.
MAX_PROVISION_JUMP = 30
UNBOUNDED_JUMP = 10_000

SUBSECTION_RE = re.compile(r"^\s*(?:\d{1,2}\s*\[\s*)?\(\s*(\d{1,2}[A-Z]?)\s*\)")
CLAUSE_RE = re.compile(r"^\s*(?:\d{1,2}\s*\[\s*)?\(\s*([a-z]{1,3})\s*\)")

#: Which locator key a provision number belongs in for this instrument.
LABEL_TO_KEY = {
    "Section": "section",
    "Rule": "rule",
    "Regulation": "regulation",
    "Paragraph": "paragraph",
    "Article": "article",
}
SUB_KEY = {
    "section": "subsection",
    "rule": "sub_rule",
    "regulation": "subsection",
    "paragraph": "subsection",
    "article": "paragraph",
}


def _is_headingish(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3:
        return False
    upper = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper >= 0.8 and len(text.strip()) <= 120


def parse(lines: list[Line], provision_label: str = "Section") -> list[Unit]:
    """Split a cleaned instrument into structural units."""
    provision_key = LABEL_TO_KEY.get(provision_label, "section")
    units: list[Unit] = []

    state: dict[str, Any] = {"part": None, "chapter": None, "schedule": None,
                             "form": None, "annexure": None}
    previous_number: tuple[int, str] | None = None
    #: The last number opened as a provision of the instrument itself, kept
    #: across Schedules and Forms. A container resets `previous_number` so its
    #: own entries can start again at 1; this one is what says whether a later
    #: number resumes the instrument's sequence or belongs to the container.
    last_provision: tuple[int, str] | None = None
    restart_used = False
    current = Unit(unit_type="front_matter", number=None, heading=None, lines=[],
                   locator=empty_locator())

    def close() -> None:
        nonlocal current
        if not current.is_empty():
            units.append(current)

    def open_unit(unit_type: str, number: str | None, heading: str | None) -> Unit:
        locator = empty_locator()
        locator["part"] = state["part"]
        locator["chapter"] = state["chapter"]
        locator["schedule"] = state["schedule"]
        locator["form"] = state["form"]
        locator["annexure"] = state["annexure"]
        if unit_type == "provision" and number is not None:
            locator[provision_key] = number
        if unit_type == "schedule_item" and number is not None:
            locator["item"] = number
        return Unit(unit_type=unit_type, number=number, heading=heading, lines=[],
                    locator=locator)

    index = 0
    while index < len(lines):
        page, text = lines[index]
        stripped = text.strip()
        if not stripped:
            current.lines.append((page, text))
            index += 1
            continue

        chapter = CHAPTER_RE.match(stripped)
        part = PART_RE.match(stripped)
        schedule = SCHEDULE_RE.match(stripped)
        form = FORM_RE.match(stripped)
        annexure = ANNEX_RE.match(stripped)

        if chapter or part:
            close()
            label = (chapter or part).group(1).upper()
            title = None
            for look in range(index + 1, min(index + 4, len(lines))):
                candidate = lines[look][1].strip()
                if candidate and _is_headingish(candidate):
                    title = re.sub(r"\s+", " ", candidate)
                    break
                if candidate:
                    break
            if chapter:
                state["chapter"] = label
            else:
                state["part"] = label
                state["chapter"] = None
            state.update({"schedule": None, "form": None, "annexure": None})
            # Section and rule numbering runs continuously across Chapters and
            # Parts, so the ascending-number guard is NOT reset here. Only a
            # Schedule, Form or Annexure restarts numbering at 1.
            current = open_unit("heading", None, title)
            current.lines.append((page, text))
            index += 1
            continue

        if schedule or annexure or form:
            close()
            previous_number = None
            if schedule:
                ordinal, letter = schedule.group(1), schedule.group(2)
                state["schedule"] = " ".join(
                    part for part in [(ordinal or "").upper(), (letter or "").upper()] if part
                ) or "SCHEDULE"
                state.update({"form": None, "annexure": None})
                unit_type = "schedule"
                number = state["schedule"]
            elif form:
                state["form"] = form.group(1).upper()
                unit_type = "form"
                number = state["form"]
            else:
                state["annexure"] = (annexure.group(1) or "ANNEXURE").upper()
                state["form"] = None
                unit_type = "annexure"
                number = state["annexure"]
            current = open_unit(unit_type, number, re.sub(r"\s+", " ", stripped))
            current.lines.append((page, text))
            index += 1
            continue

        provision = PROVISION_RE.match(stripped)
        if provision:
            candidate = parse_number(provision.group(1))
            heading = extract_heading(stripped[provision.end():])
            # A marginal heading ("55. Opposition to the patent.—") is decisive
            # evidence of a provision opener, so it lifts the bounded-jump guard.
            # That guard exists only to reject enumerated lists inside provisos,
            # which never carry headings.
            max_jump = MAX_PROVISION_JUMP if heading is None else UNBOUNDED_JUMP
            # Front matter — a contents page, a list of amending Acts — is
            # numbered exactly like sections, and consumes the numbers the real
            # body needs. One restart per document, at the first "1." that
            # carries a marginal heading, hands them back. Schedules restart
            # separately and explicitly, via SCHEDULE_RE.
            restart = (
                not restart_used
                and heading is not None
                and candidate == (1, "")
                and previous_number is not None
            )
            in_container = any(state[key] for key in ("schedule", "form", "annexure"))
            # A number that resumes the instrument's own sequence means the
            # container has ended, whether or not anything announced it. It
            # routinely does not: the Drugs and Cosmetics Rules print "Form 37."
            # at the end of rule 150E's text, on its own line, and a heading
            # match there swallowed rules 150F to 161 into a Form that had never
            # started. Resumption is the evidence that we are back in the body.
            # `last_provision is not None` matters: a container entry cannot
            # resume a sequence that has not started. Without it the first entry
            # of a Schedule appearing before any provision — a Schedule printed
            # in front matter — would be read as provision 1 of the instrument.
            resumes_body = (in_container and last_provision is not None
                            and accepts_next(last_provision, candidate,
                                             max_jump=max_jump))
            if resumes_body:
                state.update({"schedule": None, "form": None, "annexure": None})
                in_container = False
                previous_number = last_provision

            if (resumes_body or restart
                    or accepts_next(previous_number, candidate, max_jump=max_jump)):
                close()
                restart_used = restart_used or restart
                previous_number = candidate
                number = f"{candidate[0]}{candidate[1]}"
                # Inside a Schedule, Form or Annexure the numbering is the
                # container's own and restarts at 1. Entry 3 of the Patents Act
                # Schedule is an instruction to amend the 1911 Act, not section
                # 3 of the 1970 Act; giving it `section: 3` put it into direct
                # competition with the real section 3 for identifier queries.
                current = open_unit(
                    "schedule_item" if in_container else "provision", number, heading)
                if not in_container:
                    last_provision = candidate
                current.lines.append((page, text))
                index += 1
                continue

        current.lines.append((page, text))
        index += 1

    close()
    for unit in units:
        unit.structural_path = build_path(unit, provision_key)
    return units


def build_path(unit: Unit, provision_key: str) -> str:
    parts = []
    loc = unit.locator
    if loc.get("part"):
        parts.append(f"Part {loc['part']}")
    if loc.get("chapter"):
        parts.append(f"Chapter {loc['chapter']}")
    if loc.get("schedule"):
        parts.append(f"{loc['schedule']} SCHEDULE".replace("SCHEDULE SCHEDULE", "SCHEDULE"))
    if loc.get("form"):
        parts.append(f"Form {loc['form']}")
    if loc.get("annexure"):
        parts.append(f"Annexure {loc['annexure']}")
    if loc.get("item"):
        parts.append(f"Item {loc['item']}")
    elif loc.get(provision_key):
        parts.append(f"{provision_key.replace('_', ' ').title()} {loc[provision_key]}")
    return " > ".join(parts)


def subdivisions(lines: list[Line], provision_key: str) -> list[tuple[int, str, str]]:
    """Indices of sub-provision boundaries: ``(line index, kind, marker)``."""
    out: list[tuple[int, str, str]] = []
    previous: tuple[int, str] | None = None
    for index, (_, text) in enumerate(lines):
        match = SUBSECTION_RE.match(text)
        if not match:
            continue
        candidate = parse_number(match.group(1))
        if accepts_next(previous, candidate, max_jump=10):
            previous = candidate
            out.append((index, SUB_KEY.get(provision_key, "subsection"), match.group(1)))
    return out
