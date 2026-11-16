"""Phase 3 — build the provenance registry, acquisition manifest and report.

The evidence below is curated by hand and every field is sourced. Three kinds
of evidence appear, and they are never mixed:

1. **Mechanical** — a publisher copy was downloaded and compared page by page
   against the local file by `verify_source.py`. These verdicts are read back
   out of `data/metadata/source_verifications.json`, not retyped here, so the
   registry cannot drift from the comparison that produced it.
2. **Internal** — something the local PDF says about itself: a Gazette masthead,
   an India Code per-page stamp, an "[As on <date>]" line, a publisher's email.
   Verifiable by re-reading the corpus, and recorded as `local_evidence`.
3. **Documentary** — a primary instrument fetched from a publisher of record
   (chiefly egazette.gov.in) that bears on whether the local copy is current.

Anything that rests only on secondary reporting — a law-firm note, a news item,
an aggregator — is either omitted or written down as an explicit `null` with the
reason. `Never guess` is the operative instruction of §5, and an honest gap is
more useful downstream than a plausible invention.
"""
from __future__ import annotations

import json
import os
import sys

from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.common import METADATA_DIR, logical_documents  # noqa: E402
from ingest.provenance import EVIDENCE_LEVELS  # noqa: E402

REGISTRY_VERSION = "0.2.0"
RETRIEVED_AT = "2026-08-30"
VERIFICATIONS_PATH = os.path.join(METADATA_DIR, "source_verifications.json")

#: Verdicts from verify_source.py that tie the local file to a publisher copy.
MATCHED_VERDICTS = {"BYTE_IDENTICAL", "CONTENT_IDENTICAL", "CONTENT_IDENTICAL_TEXT"}

#: Whether the corpus copy still states the law as it currently stands.
#: Sourced from §3 of ``docs/PROVENANCE_REPORT.md``, which tested each document
#: against a primary instrument rather than commentary. This is a separate axis
#: from evidence level: a page-for-page verified copy of a 2016 consolidation is
#: level A and ``STALE`` at the same time, and retrieval must be able to say so.
#: ``UNKNOWN`` is reserved for documents we do not hold.
TEMPORAL_STATUS = {
    "IN-DC-RULES-1945": "STALE",       # CDSCO consolidates to G.S.R. 360(E) of 01-07-2024
    "IN-DC-ACT-1940": "STALE",         # Jan Vishwas Act, 2026 amends s. 27A
    "IN-PATENTS-RULES-2003": "STALE",  # IP India e-version updated till 15-03-2024
    "WTO-TRIPS": "STALE",              # 1994 unamended text; Article 31bis absent
    "IN-DESIGNS-RULES-2001": "STALE",  # Amendment Rules 2008, 2014, 2021
    "IN-GI-RULES-2002": "STALE",       # Amendment Rules 2020 and 2025
    "IN-COPYRIGHT-ACT-1957": "STALE",  # Jan Vishwas Act, 2026 omits s. 67
}
DEFAULT_TEMPORAL_STATUS = "CURRENT"


INDIA_CODE = "India Code, Legislative Department, Ministry of Law and Justice"
INDIA_CODE_NOTE = (
    "The local PDF carries the India Code per-page stamp on every page, which "
    "identifies the publisher beyond doubt. It could not be matched against a "
    "fetched copy because indiacode.nic.in refuses automated retrieval "
    "(handle pages 404, bitstream PDFs 302 to an error page) and indiacode.gov.in "
    "serves a JavaScript shell with no content. Verified by direct probe on "
    "2026-08-30, not merely reported."
)

# ---------------------------------------------------------------------------
# Curated evidence. `matched_url` names the publisher copy that verify_source.py
# compared against; the verdict itself is read from source_verifications.json.
# ---------------------------------------------------------------------------

SOURCES: list[dict[str, Any]] = [
    {
        "document_id": "IN-PATENTS-ACT-1970",
        "official_title": "The Patents Act, 1970 (Act No. 39 of 1970)",
        "publisher": INDIA_CODE,
        "official_source_url": "https://www.indiacode.nic.in/",
        "official_document_url": None,
        "publication_date": "1970-09-19",
        "consolidated_as_of": "2026-05-05",
        "reference_number": "Act No. 39 of 1970",
        "source_status": "SOURCE_UNAVAILABLE",
        "evidence_level": "B",
        "evidence_reason": (
            "India Code publisher identity is established from the per-page stamp "
            "and the '[As on the 5th May, 2026]' front-page line, but the "
            "repository blocks automated retrieval so no publisher copy could be "
            "fetched and compared."
        ),
        "local_evidence": "India Code stamp on all 72 pages; '[As on the 5th May, 2026]'.",
        "replacement_required": False,
        "notes": (
            "Current. The Jan Vishwas (Amendment of Provisions) Act, 2026 "
            "(Act 8 of 2026, assented 8 April 2026, fetched from egazette.gov.in) "
            "amends this Act at Serial 35 of its Schedule — Chapter XX heading "
            "'PENALTIES' to 'PUNISHMENTS', a new proviso after section 118, and "
            "section 119 omitted. The local copy already reads "
            "'CHAPTER XX 2[PUNISHMENTS]' and '119. [Omitted.]', so it incorporates "
            "that amendment. " + INDIA_CODE_NOTE
        ),
    },
    {
        "document_id": "IN-PATENTS-RULES-2003",
        "official_title": "The Patents Rules, 2003",
        "publisher": (
            "Office of the Controller General of Patents, Designs and Trade Marks "
            "(IP India), DPIIT — republished through India Code"
        ),
        "official_source_url": "https://ipindia.gov.in/pages/patents/rules-patents-2003",
        "official_document_url": None,
        "publication_date": "2003-05-02",
        "consolidated_as_of": "2017-06-23",
        "reference_number": None,
        "source_status": "OFFICIAL_SOURCE_FOUND_REPLACEMENT_AVAILABLE",
        "evidence_level": "B",
        "evidence_reason": (
            "Publisher identity established from the India Code stamp and the "
            "in-document ipindia.gov.in imprint; no fetchable copy to compare "
            "against, and the copy we hold is superseded."
        ),
        "local_evidence": (
            "India Code stamp on all 102 pages; filename and title state "
            "'Updated till 23-06-2017'; www.ipindia.gov.in appears in the text."
        ),
        "replacement_required": True,
        "notes": (
            "STALE by nine years. The IP India landing page states its e-version "
            "is 'updated till 15-03-2024'; the resolved PDF behind that page could "
            "not be captured, so the replacement is identified but not yet "
            "acquired. Reports of a Patents (Amendment) Rules, 2025 rest only on "
            "secondary sources and are recorded as unconfirmed."
        ),
    },
    {
        "document_id": "IN-TM-ACT-1999",
        "official_title": "The Trade Marks Act, 1999 (Act No. 47 of 1999)",
        "publisher": INDIA_CODE,
        "official_source_url": "https://www.indiacode.nic.in/",
        "official_document_url": None,
        "publication_date": "1999-12-30",
        "consolidated_as_of": "2026-06-01",
        "reference_number": "Act No. 47 of 1999",
        "source_status": "SOURCE_UNAVAILABLE",
        "evidence_level": "B",
        "evidence_reason": (
            "India Code identity established from the per-page stamp and the "
            "'[As on the 1st June, 2026]' line; repository unreachable for "
            "automated comparison."
        ),
        "local_evidence": "India Code stamp on all 56 pages; '[As on the 1st June, 2026]'.",
        "replacement_required": False,
        "notes": (
            "No amendment found after the stated consolidation date. The Act is "
            "absent from the Schedule to the Jan Vishwas (Amendment of Provisions) "
            "Act, 2026, checked directly against the Gazette text. " + INDIA_CODE_NOTE
        ),
    },
    {
        "document_id": "IN-TM-RULES-2017",
        "official_title": "The Trade Marks Rules, 2017",
        "publisher": "DPIIT / IP India — republished through India Code",
        "official_source_url": "https://ipindia.gov.in/trade-marks-resources-rules",
        "official_document_url": None,
        "publication_date": "2017-03-06",
        "consolidated_as_of": None,
        "reference_number": "G.S.R. 199(E)",
        "source_status": "OFFICIAL_SOURCE_FOUND_LOCAL_COPY_NOT_MATCHED",
        "evidence_level": "B",
        "evidence_reason": (
            "India Code identity established from the per-page stamp and the "
            "Gazette masthead. Five PDFs published on the IP India Trade Marks "
            "rules page were downloaded and compared; all five differ in page "
            "count from the local copy, so none is the same rendering."
        ),
        "local_evidence": (
            "India Code stamp on all 97 pages; Gazette of India masthead, "
            "'PUBLISHED BY AUTHORITY', G.S.R. 199(E)."
        ),
        "replacement_required": False,
        "local_copy_status_override": "NOT_MATCHED",
        "notes": (
            "No amendment to the 2017 Rules has been found as notified. A draft "
            "Trade Marks (1st Amendment) Rules, 2024 went out for comment; no "
            "final notification was located, and it is recorded as unconfirmed."
        ),
    },
    {
        "document_id": "IN-GI-ACT-1999",
        "official_title": (
            "The Geographical Indications of Goods (Registration and Protection) "
            "Act, 1999 (Act No. 48 of 1999)"
        ),
        "publisher": INDIA_CODE,
        "official_source_url": "https://www.indiacode.nic.in/",
        "official_document_url": None,
        "publication_date": "1999-12-30",
        "consolidated_as_of": "2026-06-01",
        "reference_number": "Act No. 48 of 1999",
        "source_status": "SOURCE_UNAVAILABLE",
        "evidence_level": "B",
        "evidence_reason": (
            "India Code identity established from the per-page stamp and the "
            "'[As on the 1st June, 2026]' line; repository unreachable."
        ),
        "local_evidence": "India Code stamp on all 31 pages; '[As on the 1st June, 2026]'.",
        "replacement_required": False,
        "notes": (
            "Absent from the Jan Vishwas 2026 Schedule. A DPIIT consultation on a "
            "GI amendment Bill was reported in 2024; no enacted amendment was "
            "found, and the Bill's status is unconfirmed. " + INDIA_CODE_NOTE
        ),
    },
    {
        "document_id": "IN-GI-RULES-2002",
        "official_title": (
            "The Geographical Indications of Goods (Registration and Protection) "
            "Rules, 2002"
        ),
        "publisher": "DPIIT / IP India",
        "official_source_url": "https://ipindia.gov.in/geographical-indications-resources-rules",
        "official_document_url": (
            "https://ipindia.gov.in/storage/uploads/docs-operator/"
            "c22590d2-e1ad-41c6-af6c-f29253810b2c.pdf"
        ),
        "matched_url": (
            "https://ipindia.gov.in/storage/uploads/docs-operator/"
            "c22590d2-e1ad-41c6-af6c-f29253810b2c.pdf"
        ),
        "publication_date": "2002-03-08",
        "consolidated_as_of": "2002-03-08",
        "reference_number": "G.S.R. 176(E)",
        "source_status": "OFFICIAL_SOURCE_FOUND_REPLACEMENT_AVAILABLE",
        "evidence_level": "A",
        "evidence_reason": (
            "All 71 pages match the IP India copy exactly once the India Code "
            "per-page stamp is removed, on text, geometry and embedded images. "
            "The local file is the India Code rendering of the document IP India "
            "publishes."
        ),
        "local_evidence": "India Code stamp on all 71 pages; Gazette citation of 8 March 2002.",
        "replacement_required": True,
        "notes": (
            "Verified but superseded: the IP India GI rules page also publishes "
            "Amendment Rules of 2020 and 2025, neither of which is in the local "
            "copy. The base 2002 text is sound; the amendments must be acquired."
        ),
    },
    {
        "document_id": "IN-DESIGNS-ACT-2000",
        "official_title": "The Designs Act, 2000 (Act No. 16 of 2000)",
        "publisher": INDIA_CODE,
        "official_source_url": "https://www.indiacode.nic.in/",
        "official_document_url": None,
        "publication_date": "2000-05-25",
        "consolidated_as_of": "2026-06-15",
        "reference_number": "Act No. 16 of 2000",
        "source_status": "SOURCE_UNAVAILABLE",
        "evidence_level": "B",
        "evidence_reason": (
            "India Code identity established from the per-page stamp and the "
            "'[As on the 15th June, 2026]' line; repository unreachable."
        ),
        "local_evidence": "India Code stamp on all 18 pages; '[As on the 15th June, 2026]'.",
        "replacement_required": False,
        "notes": "Absent from the Jan Vishwas 2026 Schedule. " + INDIA_CODE_NOTE,
    },
    {
        "document_id": "IN-DESIGNS-RULES-2001",
        "official_title": "The Designs Rules, 2001",
        "publisher": "DPIIT / IP India",
        "official_source_url": "https://ipindia.gov.in/designs-resources-rules",
        "official_document_url": (
            "https://ipindia.gov.in/storage/uploads/docs-operator/"
            "87a7da10-66ad-4d75-a9ff-60f2fde67143.pdf"
        ),
        "matched_url": (
            "https://ipindia.gov.in/storage/uploads/docs-operator/"
            "87a7da10-66ad-4d75-a9ff-60f2fde67143.pdf"
        ),
        "publication_date": "2001-05-11",
        "consolidated_as_of": "2001-05-11",
        "reference_number": "S.O. 414(E)",
        "source_status": "OFFICIAL_SOURCE_FOUND_REPLACEMENT_AVAILABLE",
        "evidence_level": "A",
        "evidence_reason": (
            "All 31 pages match the IP India copy on text, geometry and embedded "
            "images."
        ),
        "local_evidence": "Notification of 11 May 2001, S.O. 414(E).",
        "replacement_required": True,
        "notes": (
            "Verified as the original 2001 Rules and therefore superseded: IP "
            "India separately publishes Designs (Amendment) Rules of 2008, 2014 "
            "and 2021, and a consolidated e-version incorporating amendments "
            "through 25 January 2021. The local copy predates all three."
        ),
    },
    {
        "document_id": "IN-COPYRIGHT-ACT-1957",
        "official_title": "The Copyright Act, 1957 (Act No. 14 of 1957)",
        "publisher": (
            "Copyright Office, Department for Promotion of Industry and Internal "
            "Trade, Ministry of Commerce and Industry"
        ),
        "official_source_url": "https://copyright.gov.in/",
        "official_document_url": "https://copyright.gov.in/Documents/Copyright_Act_1957.pdf",
        "matched_url": "https://copyright.gov.in/Documents/Copyright_Act_1957.pdf",
        "publication_date": "1957-06-04",
        "consolidated_as_of": None,
        "reference_number": "Act No. 14 of 1957",
        "source_status": "OFFICIAL_SOURCE_FOUND_LOCAL_COPY_NOT_MATCHED",
        "evidence_level": "C",
        "evidence_reason": (
            "The local file carries no publisher imprint of any kind — no India "
            "Code stamp, no Gazette masthead, no 'As on' line — and it does not "
            "match the Copyright Office copy, which runs to 52 pages against the "
            "local 57. Two different consolidations of the same Act; which "
            "publisher produced the local one is unresolved."
        ),
        "local_evidence": "None. The document states only its title and Act number.",
        "replacement_required": True,
        "notes": (
            "Both copies predate a passed amendment. The Jan Vishwas (Amendment "
            "of Provisions) Act, 2026, Serial 23 of its Schedule, omits section "
            "67; section 67 ('Penalty for making false entries in register') is "
            "still present in full in the local copy AND in the Copyright Office "
            "copy. Commencement is by notification under section 1(2) and no "
            "commencement notification was located, so the amendment's in-force "
            "date is unconfirmed. Replace with the Copyright Office copy for "
            "provenance, and re-check once a post-commencement consolidation "
            "appears."
        ),
    },
    {
        "document_id": "IN-COPYRIGHT-RULES-2013",
        "official_title": "The Copyright Rules, 2013",
        "publisher": "Copyright Office, DPIIT",
        "official_source_url": "https://copyright.gov.in/",
        "official_document_url": (
            "https://copyright.gov.in/Documents/Copyright_Rules_2013_and_Forms.pdf"
        ),
        "matched_url": (
            "https://copyright.gov.in/Documents/Copyright_Rules_2013_and_Forms.pdf"
        ),
        "publication_date": "2013-03-14",
        "consolidated_as_of": None,
        "reference_number": "G.S.R. 172(E)",
        "source_status": "VERIFIED_OFFICIAL",
        "evidence_level": "A",
        "evidence_reason": (
            "All 77 pages match the Copyright Office copy on text, geometry and "
            "embedded images."
        ),
        "local_evidence": "copyright.gov.in appears in the document text.",
        "replacement_required": False,
        "notes": (
            "The local copy is the file the Copyright Office serves today. A "
            "Copyright (Amendment) Rules, 2021 exists and is published separately "
            "by IP India; it is not incorporated here and should be acquired as a "
            "supplementary instrument. Its G.S.R. number is disputed between "
            "225(E) and 226(E) across secondary sources and is recorded as "
            "unresolved — no primary gazette copy was obtained."
        ),
    },
    {
        "document_id": "IN-BDA-2002-AMD-2023",
        "official_title": "The Biological Diversity Act, 2002 (Act No. 18 of 2003)",
        "publisher": None,
        "official_source_url": "https://www.nbaindia.nic.in/",
        "official_document_url": None,
        "publication_date": "2003-02-05",
        "consolidated_as_of": "2024-04-01",
        "reference_number": "Act No. 18 of 2003",
        "source_status": "PROVENANCE_UNVERIFIED",
        "evidence_level": "C",
        "evidence_reason": (
            "A credible consolidated text with no publisher imprint: no India "
            "Code stamp, no Gazette masthead, no 'As on' line. The NBA's "
            "historical deep links now redirect to its homepage and no copy could "
            "be fetched for comparison."
        ),
        "local_evidence": "None beyond the title and arrangement of sections.",
        "replacement_required": True,
        "notes": (
            "The amending Act the local copy claims to incorporate was confirmed "
            "directly from the Gazette: the Biological Diversity (Amendment) Act, "
            "2023 is Act 10 of 2023, assented 3 August 2023 (egazette.gov.in). "
            "The consolidation looks right; only its publisher is unknown. "
            "nbaindia.org now redirects to nbaindia.nic.in and the old PDF paths "
            "are dead — re-acquire through the NBA site's own navigation."
        ),
    },
    {
        "document_id": "IN-BD-RULES-2024",
        "official_title": "The Biological Diversity Rules, 2024",
        "publisher": (
            "Ministry of Environment, Forest and Climate Change — republished "
            "through India Code"
        ),
        "official_source_url": "https://www.indiacode.nic.in/",
        "official_document_url": None,
        "publication_date": None,
        "consolidated_as_of": None,
        "reference_number": "G.S.R. 665(E)",
        "source_status": "SOURCE_UNAVAILABLE",
        "evidence_level": "B",
        "evidence_reason": (
            "India Code identity established from the per-page stamp on all 86 "
            "pages, plus a Gazette masthead and 'PUBLISHED BY AUTHORITY' in the "
            "document. No MoEFCC or India Code copy could be fetched."
        ),
        "local_evidence": (
            "India Code stamp on all 86 pages; Gazette of India masthead; "
            "G.S.R. 665(E) as printed in the document."
        ),
        "replacement_required": False,
        "notes": (
            "The notification date could not be confirmed from a publisher of "
            "record and is left null: secondary sources give 22, 25 and 28 "
            "October 2024 inconsistently, and MoEFCC deep links return 404. The "
            "G.S.R. number here is taken from the local document itself, not from "
            "secondary reporting. These Rules supersede the Biological Diversity "
            "Rules, 2004."
        ),
    },
    {
        "document_id": "IN-ABS-REGULATIONS-2025",
        "official_title": (
            "Biological Diversity (Access to Biological Resources and Associated "
            "Knowledge and Fair and Equitable Sharing of Benefits) Regulations, 2025"
        ),
        "publisher": "National Biodiversity Authority, Chennai",
        "official_source_url": "https://www.nbaindia.nic.in/",
        "official_document_url": None,
        "publication_date": "2025-04-29",
        "consolidated_as_of": "2025-04-29",
        "reference_number": "F. No. NBA/Tech/EC/9/14/32",
        "source_status": "SOURCE_UNAVAILABLE",
        "evidence_level": "B",
        "evidence_reason": (
            "The document carries a full Gazette of India Extraordinary masthead "
            "and 'PUBLISHED BY AUTHORITY', which establishes it as a gazette "
            "copy. The NBA's own PDF paths are dead: nbaindia.org redirects to "
            "the homepage and nbaindia.nic.in returns 404, both probed directly."
        ),
        "local_evidence": (
            "Gazette of India Extraordinary Part III Sec. 4 masthead; NBA "
            "notification, Chennai, 29 April 2025; F. No. NBA/Tech/EC/9/14/32."
        ),
        "replacement_required": False,
        "notes": (
            "Current instrument; replaces the 2014 ABS Guidelines. Only the "
            "publisher's own hosted copy is missing, not the document's identity."
        ),
    },
    {
        "document_id": "IN-PPVFR-ACT-2001",
        "official_title": (
            "The Protection of Plant Varieties and Farmers' Rights Act, 2001 "
            "(Act No. 53 of 2001)"
        ),
        "publisher": (
            "Protection of Plant Varieties and Farmers' Rights Authority, "
            "Ministry of Agriculture and Farmers' Welfare"
        ),
        "official_source_url": (
            "https://plantauthority.gov.in/protection-plant-varieties-and-farmers-rights-act-2001"
        ),
        "official_document_url": "https://plantauthority.gov.in/sites/default/files/ppvfract2001.pdf",
        "matched_url": "https://plantauthority.gov.in/sites/default/files/ppvfract2001.pdf",
        "publication_date": "2001-10-30",
        "consolidated_as_of": "2001-10-30",
        "reference_number": "Act No. 53 of 2001",
        "source_status": "VERIFIED_OFFICIAL",
        "evidence_level": "A",
        "evidence_reason": (
            "All 50 pages match the PPV&FR Authority copy on text, geometry and "
            "embedded images."
        ),
        "local_evidence": (
            "Gazette of India Extraordinary masthead, REGISTERED NO. DL-33004/2001, "
            "Part II Section 1, No. 64, 30 October 2001, Ministry of Law, Justice "
            "and Company Affairs."
        ),
        "replacement_required": False,
        "notes": (
            "Phase 2 recorded no Act number for this document; the Gazette text "
            "establishes it as Act 53 of 2001, now recorded. Original enactment, "
            "not a periodically refreshed consolidation."
        ),
    },
    {
        "document_id": "IN-PPVFR-RULES-2003",
        "official_title": (
            "The Protection of Plant Varieties and Farmers' Rights Rules, 2003"
        ),
        "publisher": "PPV&FR Authority, Ministry of Agriculture and Farmers' Welfare",
        "official_source_url": "https://plantauthority.gov.in/acts-rules",
        "official_document_url": "https://plantauthority.gov.in/sites/default/files/ppvfrarules2003.pdf",
        "matched_url": "https://plantauthority.gov.in/sites/default/files/ppvfrarules2003.pdf",
        "publication_date": "2003-09-12",
        "consolidated_as_of": None,
        "reference_number": None,
        "source_status": "VERIFIED_OFFICIAL",
        "evidence_level": "A",
        "evidence_reason": (
            "All 56 pages match the PPV&FR Authority copy on text, geometry and "
            "embedded images."
        ),
        "local_evidence": "Notification of 12 September 2003.",
        "replacement_required": False,
        "notes": (
            "The local copy is the file the Authority serves today. Amendments to "
            "the Rules are reported for 2004 and 2008 from secondary sources only "
            "and are recorded as unconfirmed. A 2025 instrument found in search is "
            "an amendment to the Authority's Regulations, a different instrument "
            "from these Rules, and must not be conflated with them."
        ),
    },
    {
        "document_id": "IN-DC-ACT-1940",
        "official_title": "The Drugs and Cosmetics Act, 1940 (Act 23 of 1940)",
        "publisher": (
            "Central Drugs Standard Control Organisation, Directorate General of "
            "Health Services, Ministry of Health and Family Welfare"
        ),
        "official_source_url": "https://cdsco.gov.in/opencms/opencms/en/Acts-and-rules/",
        "official_document_url": (
            "https://cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/"
            "acts_rules/2016DrugsandCosmeticsAct1940Rules1945.pdf"
        ),
        "matched_url": (
            "https://cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/"
            "acts_rules/2016DrugsandCosmeticsAct1940Rules1945.pdf"
        ),
        "publication_date": "1940-04-10",
        "consolidated_as_of": "2016-12-31",
        "reference_number": "Act 23 of 1940",
        "source_status": "OFFICIAL_SOURCE_FOUND_REPLACEMENT_AVAILABLE",
        "evidence_level": "A",
        "evidence_reason": (
            "All 635 pages of the compendium match the CDSCO copy on text, "
            "geometry and embedded images. Both logical documents in this "
            "compendium inherit the verdict from the same physical file."
        ),
        "local_evidence": (
            "Title page names the Ministry of Health and Family Welfare and states "
            "'as amended up to 31st December 2016'."
        ),
        "replacement_required": True,
        "notes": (
            "STALE by nine years, and amended since: the Jan Vishwas (Amendment "
            "of Provisions) Act, 2026 amends section 27A of this Act (confirmed "
            "in the Gazette text at Serial 22 of its Schedule). CDSCO publishes "
            "the Act separately, released 22 November 2022."
        ),
    },
    {
        "document_id": "IN-DC-RULES-1945",
        "official_title": "The Drugs Rules, 1945",
        "publisher": "CDSCO, Ministry of Health and Family Welfare",
        "official_source_url": "https://cdsco.gov.in/opencms/opencms/en/Acts-and-rules/Drugs-Rules/",
        "official_document_url": (
            "https://cdsco.gov.in/opencms/resources/UploadCDSCOWeb/2022/drug_rules/"
            "Drugs%20Rules%201945_2024%2009.pdf"
        ),
        "matched_url": (
            "https://cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/"
            "acts_rules/2016DrugsandCosmeticsAct1940Rules1945.pdf"
        ),
        "publication_date": "1945-12-21",
        "consolidated_as_of": "2016-12-31",
        "reference_number": "Notification No. F. 28-10/45-H(1)",
        "source_status": "OFFICIAL_SOURCE_FOUND_REPLACEMENT_AVAILABLE",
        "evidence_level": "A",
        "evidence_reason": (
            "Verified through the same physical compendium as IN-DC-ACT-1940: all "
            "635 pages match the CDSCO copy."
        ),
        "local_evidence": "Compendium title page, 'as amended up to 31st December 2016'.",
        "replacement_required": True,
        "notes": (
            "STALE by eight years, and the replacement has been acquired and its "
            "currency read from the document itself: the CDSCO consolidated Drugs "
            "Rules, 1945 runs to 963 pages and its front matter states "
            "'[As amended vide G.S.R. 360(E) dated 01-07-2024 (w.e.f. "
            "01-07-2024)]'. That is 366 pages more than the local Rules. Note also "
            "that the New Drugs and Clinical Trials Rules, 2019 and the Cosmetics "
            "Rules, 2020 have since carved subject matter out of the 1945 Rules."
        ),
    },
    {
        "document_id": "IN-FSSAI-AYURVEDA-AAHARA-2022",
        "official_title": (
            "FSSAI Order dated 25 July 2025 — Ayurveda Aahara covered under the "
            "Food Safety and Standards (Ayurveda Aahara) Regulations, 2022, "
            "with enclosures"
        ),
        "publisher": (
            "Food Safety and Standards Authority of India, Science and Standards "
            "Division, Ministry of Health and Family Welfare"
        ),
        "official_source_url": "https://www.fssai.gov.in/",
        "official_document_url": (
            "https://www.fssai.gov.in/upload/advisories/2025/07/"
            "68835f872eaf4Order%20dated%2025-07-2025%20enclosing%20Ayurveda%20Aahara.pdf"
        ),
        "matched_url": (
            "https://www.fssai.gov.in/upload/advisories/2025/07/"
            "68835f872eaf4Order%20dated%2025-07-2025%20enclosing%20Ayurveda%20Aahara.pdf"
        ),
        "publication_date": "2025-07-25",
        "consolidated_as_of": "2025-07-25",
        "reference_number": "SS-T0SP05(EC)/2/2022-Standard-FSSAI I/29912/2025",
        "source_status": "VERIFIED_OFFICIAL",
        "evidence_level": "A",
        "evidence_reason": (
            "All 172 pages match the FSSAI-hosted copy on text, geometry and "
            "embedded images. The local file is a PDFium re-encoding of the "
            "publisher's own PDF (whose ModDate is 25 July 2025)."
        ),
        "local_evidence": (
            "FSSAI letterhead, Science and Standards Division, "
            "ayurveda-aahara@fssai.gov.in, dated 25 July 2025."
        ),
        "replacement_required": False,
        "content_matches_expected_title": False,
        "notes": (
            "VERIFIED AS WHAT IT IS, NOT AS WHAT IT WAS FILED UNDER. This file is "
            "the FSSAI Order of 25 July 2025 plus a 170-page scanned index of "
            "Ayurveda Aahara preparations and their textual references. It is not "
            "the Regulations. Its filename in the corpus asserts otherwise and "
            "must not be trusted; the Regulations are registered separately as "
            "IN-FSSAI-AYURVEDA-AAHARA-REGS-2022, status MISSING. Confirmed again "
            "in this phase: the FSSAI 'Ayurveda Aahara Compendium' (647 pages, "
            "fetched from fssai.gov.in) is likewise a recipe index containing no "
            "regulatory prose."
        ),
    },
    {
        "document_id": "IN-NATIONAL-IPR-POLICY-2016",
        "official_title": (
            "National Intellectual Property Rights Policy — 'Creative India; "
            "Innovative India'"
        ),
        "publisher": (
            "Department for Promotion of Industry and Internal Trade, Ministry of "
            "Commerce and Industry"
        ),
        "official_source_url": "https://cipam.gov.in/en/about/national-IPR-policy",
        "official_document_url": (
            "https://www.dpiit.gov.in/static/uploads/2025/07/"
            "442648f225695b31b259e9d9edd0a8b1.pdf"
        ),
        "matched_url": (
            "https://www.dpiit.gov.in/static/uploads/2025/07/"
            "442648f225695b31b259e9d9edd0a8b1.pdf"
        ),
        "publication_date": "2016-05-12",
        "consolidated_as_of": None,
        "reference_number": None,
        "source_status": "OFFICIAL_SOURCE_FOUND_LOCAL_COPY_NOT_MATCHED",
        "evidence_level": "B",
        "evidence_reason": (
            "The DPIIT copy has the same 36 pages and the same cover, but is an "
            "OCR scan (Adobe Paper Capture) while the local file carries a native "
            "text layer, so only 9 of 36 text layers compare equal. Same document, "
            "two different digitisations — identity established, copy not matched."
        ),
        "local_evidence": "DPIIT named on the cover; 2020 print edition dated 03/02/2020.",
        "replacement_required": False,
        "notes": (
            "A policy document, not legislation. No supersession was found; the "
            "policy's own five-yearly review requirement fell due around 2021 and "
            "no revised edition was located. Prefer the local native-text copy "
            "over the DPIIT OCR scan for retrieval quality."
        ),
    },
    {
        "document_id": "WTO-TRIPS",
        "official_title": (
            "Agreement on Trade-Related Aspects of Intellectual Property Rights "
            "(Annex 1C to the Marrakesh Agreement Establishing the World Trade "
            "Organization)"
        ),
        "publisher": "World Trade Organization",
        "official_source_url": "https://www.wto.org/english/docs_e/legal_e/31bis_trips_01_e.htm",
        "official_document_url": "https://www.wto.org/english/docs_e/legal_e/31bis_trips_e.pdf",
        "matched_url": "https://www.wto.org/english/docs_e/legal_e/27-trips.pdf",
        "publication_date": "1994-04-15",
        "consolidated_as_of": "1994-04-15",
        "reference_number": "Annex 1C",
        "source_status": "OFFICIAL_SOURCE_FOUND_REPLACEMENT_AVAILABLE",
        "evidence_level": "B",
        "evidence_reason": (
            "WTO authorship is established from the document's own footnote, "
            "which is the WTO's standard amendment note. Two WTO PDFs were "
            "downloaded and neither matches on page count (local 31; WTO "
            "unamended 33; WTO amended 36), so the exact WTO rendering the local "
            "file came from was not identified."
        ),
        "local_evidence": (
            "Footnote a: 'The TRIPS Agreement was amended through the Protocol "
            "Amending the TRIPS Agreement, done at Geneva on 6 December 2005, "
            "which entered into force on 23 January 2017... This file reproduces "
            "the 1994 unamended version.'"
        ),
        "replacement_required": True,
        "notes": (
            "SUBSTANTIVELY OUT OF DATE, and this is the finding that shows why "
            "'treaties do not go stale' is the wrong rule. The local copy states "
            "on its own face that it is the 1994 unamended text. It therefore "
            "lacks Article 31bis and the Annex and Appendix inserted by the 2005 "
            "Protocol, which entered into force on 23 January 2017 — the "
            "compulsory-licensing-for-export provisions most relevant to public "
            "health and pharmaceutical patents. The WTO now serves the amended "
            "text as the default. Whether India has accepted the Protocol was not "
            "confirmed from a WTO source in this phase and must be established "
            "before the amended text is treated as binding on India."
        ),
    },
    {
        "document_id": "UN-CBD-1992",
        "official_title": "Convention on Biological Diversity — Text and Annexes",
        "publisher": "Secretariat of the Convention on Biological Diversity",
        "official_source_url": "https://www.cbd.int/convention/text/",
        "official_document_url": "https://www.cbd.int/doc/legal/cbd-en.pdf",
        "matched_url": "https://www.cbd.int/doc/legal/cbd-en.pdf",
        "publication_date": "1992-06-05",
        "consolidated_as_of": "1992-06-05",
        "reference_number": None,
        "source_status": "VERIFIED_OFFICIAL",
        "evidence_level": "A",
        "evidence_reason": (
            "All 36 text layers and all 36 page geometries match the CBD "
            "Secretariat copy. One sampled page — the cover — differs only in its "
            "embedded image, which the local file's PDFium re-encoding "
            "recompressed. No word of the treaty differs."
        ),
        "local_evidence": "www.cbd.int appears in the document.",
        "replacement_required": False,
        "notes": (
            "Genuinely static: the Cartagena and Nagoya Protocols are separate "
            "supplementary instruments, not amendments to this text. Entered into "
            "force 29 December 1993."
        ),
    },
    {
        "document_id": "UN-NAGOYA-PROTOCOL-2010",
        "official_title": (
            "Nagoya Protocol on Access to Genetic Resources and the Fair and "
            "Equitable Sharing of Benefits Arising from their Utilization to the "
            "Convention on Biological Diversity"
        ),
        "publisher": "Secretariat of the Convention on Biological Diversity",
        "official_source_url": "https://www.cbd.int/abs/",
        "official_document_url": "https://www.cbd.int/abs/doc/protocol/nagoya-protocol-en.pdf",
        "matched_url": "https://www.cbd.int/abs/doc/protocol/nagoya-protocol-en.pdf",
        "publication_date": "2010-10-29",
        "consolidated_as_of": "2010-10-29",
        "reference_number": None,
        "source_status": "VERIFIED_OFFICIAL",
        "evidence_level": "A",
        "evidence_reason": (
            "All 15 pages match the CBD Secretariat copy on text, geometry and "
            "embedded images."
        ),
        "local_evidence": "www.cbd.int appears in the document.",
        "replacement_required": False,
        "notes": "Static treaty text; entered into force 12 October 2014.",
    },
]

#: Documents the corpus is expected to contain but does not.
MISSING: list[dict[str, Any]] = [
    {
        "document_id": "IN-FSSAI-AYURVEDA-AAHARA-REGS-2022",
        "official_title": (
            "Food Safety and Standards (Ayurveda Aahara) Regulations, 2022"
        ),
        "publisher": (
            "Food Safety and Standards Authority of India, Ministry of Health and "
            "Family Welfare"
        ),
        "official_source_url": "https://www.fssai.gov.in/",
        "official_document_url": None,
        "publication_date": "2022-05-05",
        "consolidated_as_of": None,
        "reference_number": "F. No. Stds/SP-05/A-1.Y(01)",
        "source_status": "MISSING",
        "evidence_level": "D",
        "evidence_reason": (
            "No publisher copy of the Regulations was obtained. The only text of "
            "the Regulations available to the project is a commercial legal "
            "database rendering, which §2 excludes as a source of record."
        ),
        "local_evidence": None,
        "replacement_required": True,
        "notes": (
            "The most Ayurveda-specific instrument on the expected list, and the "
            "corpus does not contain it. What the corpus has instead is the FSSAI "
            "covering Order of 25 July 2025 (registered as "
            "IN-FSSAI-AYURVEDA-AAHARA-2022) whose filename claims to be these "
            "Regulations. Made under section 92(2)(v) of the Food Safety and "
            "Standards Act, 2006; draft published 30 June 2021, gazette copies "
            "released to the public 5 July 2021, final notification dated 5 May "
            "2022. The FSSAI gazette URL located in this phase serves the site's "
            "JavaScript shell rather than a PDF, and the FSSAI 'Ayurveda Aahara "
            "Compendium' turns out to be another recipe index. Acquisition "
            "probably needs a browser session against the FSSAI portal or a "
            "request to FSSAI directly."
        ),
    },
]

# Candidate copies held in data/source_updates/ that are NOT of record.
CANDIDATES: list[dict[str, Any]] = [
    {
        "for_document_id": "IN-FSSAI-AYURVEDA-AAHARA-REGS-2022",
        "staged_file": (
            "data/source_updates/a690fc8c03db07c1-fssai-ayurveda-aahara-"
            "regulations-2022-legitquest.pdf"
        ),
        "sha256": "a690fc8c03db07c1446fd4532b4ff93d3a2e49130e14750776849008a51d2980",
        "origin": "LegitQuest (commercial legal database)",
        "evidence_level": "C",
        "accepted": False,
        "assessment": (
            "Contains the right text: Regulations 1 to 14 and Schedules A, B, C "
            "and D of the Food Safety and Standards (Ayurveda Aahara) "
            "Regulations, 2022, including the definition of Ayurveda Aahara, the "
            "Schedule A list of 71 authoritative books and the Schedule B "
            "category table. Rejected as a source of record on two grounds. "
            "First, provenance: the file is an mPDF/Laravel print-to-PDF stamped "
            "'Printed For: 29-08-2026' and carrying the LegitQuest logo on every "
            "page — a commercial database, which §2 excludes. Second, "
            "completeness: it ends inside Schedule D, so Schedule E (the AYURVEDA "
            "AAHARA logo, which regulation 13(a) makes mandatory on every label) "
            "is referenced but never reproduced, and there is no gazette masthead "
            "or signature block. Useful as a cross-check while the official text "
            "is sought; not admissible as the citable source."
        ),
    },
]


def load_verifications() -> dict[str, dict[str, Any]]:
    if not os.path.exists(VERIFICATIONS_PATH):
        return {}
    with open(VERIFICATIONS_PATH, encoding="utf-8") as fh:
        return {
            f"{r['document_id']}::{r['candidate_url']}": r
            for r in json.load(fh)["verifications"]
        }


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    documents = {d.document_id: d for d in logical_documents()}
    verifications = load_verifications()
    curated = {entry["document_id"]: entry for entry in SOURCES}

    missing_from_curation = set(documents) - set(curated)
    if missing_from_curation:
        raise SystemExit(f"no curated evidence for: {sorted(missing_from_curation)}")

    registry_sources: list[dict[str, Any]] = []
    for document_id, document in documents.items():
        entry = curated[document_id]
        matched_url = entry.get("matched_url") or entry.get("official_document_url")
        verification = verifications.get(f"{document_id}::{matched_url}") if matched_url else None
        verdict = verification["verdict"] if verification else None
        matched = verdict in MATCHED_VERDICTS

        registry_sources.append({
            "document_id": document_id,
            "local_source_filename": document.source_filename,
            "local_sha256": document.source_sha256,
            "parent_source_document_id": document.parent_source_document_id,
            "official_title": entry["official_title"],
            "publisher": entry["publisher"],
            "official_source_url": entry["official_source_url"],
            "official_document_url": entry["official_document_url"],
            "publication_date": entry["publication_date"],
            "version": document.version,
            "effective_from": entry["publication_date"],
            "effective_to": None,
            "consolidated_as_of": entry["consolidated_as_of"],
            "amendment_date": entry["consolidated_as_of"],
            "temporal_status": TEMPORAL_STATUS.get(document_id, DEFAULT_TEMPORAL_STATUS),
            "reference_number": entry["reference_number"],
            "source_version_id": (
                f"{document_id}@{entry['consolidated_as_of']}"
                if entry["consolidated_as_of"] else f"{document_id}@undated"
            ),
            "retrieved_at": RETRIEVED_AT,
            "source_status": entry["source_status"],
            "local_copy_status": entry.get("local_copy_status_override") or (
                "MATCHED" if matched else
                "NOT_MATCHED" if verdict else "NOT_COMPARED"
            ),
            "comparison": {
                "candidate_url": matched_url,
                "verdict": verdict,
                "candidate_sha256": verification.get("candidate_sha256") if verification else None,
                "pages_compared": verification.get("local_pages") if verification else None,
                "pages_with_identical_text": (
                    verification.get("pages_with_identical_text") if verification else None
                ),
            } if matched_url else None,
            "source_evidence_level": entry["evidence_level"],
            "source_evidence_level_reason": entry["evidence_reason"],
            "local_evidence": entry["local_evidence"],
            "content_matches_expected_title": entry.get("content_matches_expected_title", True),
            "replacement_required": entry["replacement_required"],
            "notes": entry["notes"],
        })

    for entry in MISSING:
        registry_sources.append({
            "document_id": entry["document_id"],
            "local_source_filename": None,
            "local_sha256": None,
            "parent_source_document_id": None,
            "official_title": entry["official_title"],
            "publisher": entry["publisher"],
            "official_source_url": entry["official_source_url"],
            "official_document_url": entry["official_document_url"],
            "publication_date": entry["publication_date"],
            "version": None,
            "effective_from": entry["publication_date"],
            "effective_to": None,
            "consolidated_as_of": entry["consolidated_as_of"],
            "amendment_date": None,
            "temporal_status": "UNKNOWN",
            "reference_number": entry["reference_number"],
            "source_version_id": None,
            "retrieved_at": RETRIEVED_AT,
            "source_status": "MISSING",
            "local_copy_status": "ABSENT",
            "comparison": None,
            "source_evidence_level": entry["evidence_level"],
            "source_evidence_level_reason": entry["evidence_reason"],
            "local_evidence": entry["local_evidence"],
            "content_matches_expected_title": False,
            "replacement_required": True,
            "notes": entry["notes"],
        })

    registry = {
        "registry_version": REGISTRY_VERSION,
        "verified_at": RETRIEVED_AT,
        "evidence_levels": EVIDENCE_LEVELS,
        "method": (
            "A local PDF is treated as matched when a copy fetched from the "
            "publisher of record agrees with it page for page on extracted text, "
            "page geometry and embedded-image digests, after removing repository "
            "viewer stamps. SHA-256 equality is reported when it occurs but is not "
            "required: fifteen of the twenty-one corpus PDFs are browser "
            "re-encodings whose container hash can never match the publisher's."
        ),
        "sources": registry_sources,
    }

    # --- acquisition manifest -------------------------------------------------
    manifest_entries = []
    for entry in registry_sources:
        curated_entry = curated.get(entry["document_id"])
        if entry["source_status"] == "MISSING":
            existing = {
                "filename": None,
                "sha256": None,
                "status": "ABSENT_FROM_CORPUS",
            }
            recommended = {
                "title": entry["official_title"],
                "official_url": entry["official_source_url"],
                "official_document_url": entry["official_document_url"],
                "publisher": entry["publisher"],
                "status": "MISSING_AND_REQUIRED",
            }
            priority = 1
        elif entry["replacement_required"]:
            existing = {
                "filename": entry["local_source_filename"],
                "sha256": entry["local_sha256"],
                "status": (
                    "WRONG_CONTENT_FOR_EXPECTED_TITLE"
                    if not entry["content_matches_expected_title"]
                    else "SUPERSEDED" if entry["consolidated_as_of"] else "PROVENANCE_UNRESOLVED"
                ),
            }
            recommended = {
                "title": entry["official_title"],
                "official_url": entry["official_source_url"],
                "official_document_url": entry["official_document_url"],
                "publisher": entry["publisher"],
                "status": "REPLACEMENT_AVAILABLE" if entry["official_document_url"]
                          else "REPLACEMENT_IDENTIFIED_URL_UNRESOLVED",
            }
            priority = 2
        else:
            existing = {
                "filename": entry["local_source_filename"],
                "sha256": entry["local_sha256"],
                "status": "ACCEPTED_FOR_USE",
            }
            recommended = {
                "title": entry["official_title"],
                "official_url": entry["official_source_url"],
                "official_document_url": entry["official_document_url"],
                "publisher": entry["publisher"],
                "status": "NO_ACTION_REQUIRED",
            }
            priority = 3

        manifest_entries.append({
            "document_id": entry["document_id"],
            "acquisition_priority": priority,
            "source_evidence_level": entry["source_evidence_level"],
            "existing_source": existing,
            "recommended_source": recommended,
            "reason": (curated_entry or {}).get("notes")
                      or next(m["notes"] for m in MISSING
                              if m["document_id"] == entry["document_id"]),
        })

    manifest = {
        "manifest_version": REGISTRY_VERSION,
        "generated_at": RETRIEVED_AT,
        "policy": (
            "No file in ~/Downloads/IP-SAKTI Corpus is ever replaced in place. "
            "Acquired copies land in data/source_updates/ under content-addressed "
            "names and supersede a local source only after an explicit approval "
            "step and a fresh ingestion run."
        ),
        "priority_key": {
            "1": "missing and required",
            "2": "replacement required (stale, wrong content, or unresolved provenance)",
            "3": "accepted for use",
        },
        "candidates_held_but_not_accepted": CANDIDATES,
        "documents": sorted(manifest_entries,
                            key=lambda e: (e["acquisition_priority"], e["document_id"])),
    }
    return registry, manifest


def main() -> int:
    registry, manifest = build()
    os.makedirs(METADATA_DIR, exist_ok=True)
    for name, payload in (
        ("provenance_registry.json", registry),
        ("source_acquisition_manifest.json", manifest),
    ):
        path = os.path.join(METADATA_DIR, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"wrote {path}")

    levels = {}
    for entry in registry["sources"]:
        levels[entry["source_evidence_level"]] = levels.get(entry["source_evidence_level"], 0) + 1
    print("evidence levels:", dict(sorted(levels.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
