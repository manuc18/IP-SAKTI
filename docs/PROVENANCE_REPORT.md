# IP-SAKTI — Phase 3: source provenance and corpus completeness

**Date:** 30 August 2026 · **Registry:** `data/metadata/provenance_registry.json` v0.1.0
**Companion artefacts:** `data/metadata/source_acquisition_manifest.json`, `data/metadata/source_verifications.json`
**Corpus:** unmodified. No file in `~/Downloads/IP-SAKTI Corpus` was renamed, replaced, overwritten or deleted; every source PDF still hashes to its Phase 1 digest.

---

## 0. Method — what "verified" means here

Phase 2 left every chunk marked `PROVENANCE_UNVERIFIED`, on the reasoning that fifteen of the twenty-one PDFs are browser re-encodings whose publisher metadata was destroyed. That reasoning was right about the metadata and wrong about the conclusion. **A container hash is not the only way to tie a file to its publisher.**

Sixteen candidate copies were downloaded from publishers of record and compared against the local files page by page, on three axes that survive re-encoding: the extracted text layer of every page, the page geometry of every page, and the SHA-256 of the embedded image XObjects. A file whose every page matches the publisher's copy on all three is the publisher's document, whatever its container hash says.

One refinement was needed. India Code prints a per-page `IndiaCode` stamp on everything it serves, which made the GI Rules look like a different document from IP India's copy. Removing that stamp, **all 71 pages match exactly** — so the local file is the India Code rendering of the document IP India publishes. The verifier now normalises repository viewer stamps before comparing, and records that it did.

Evidence levels (§10) map onto this as follows. They describe *source provenance only* — never legal correctness, and never whether the text is up to date.

| Level | Meaning here |
|---|---|
| **A** | A publisher-of-record copy was fetched in this phase and the local file matches it page for page. |
| **B** | Publisher identity is established — from a Gazette masthead, an India Code stamp, or an official URL — but no fetched copy could be matched to the local file. |
| **C** | A credible copy carrying no publisher imprint, which no fetched copy matched. |
| **D** | Unverified, or absent from the corpus. |

---

## 1. Corpus completeness

| | Count |
|---|---:|
| Logical documents registered | 22 |
| Physical source PDFs | 21 |
| Expected documents present and correctly identified | **20 of 21** |
| Expected documents present but **not the document the filename claims** | **1** |
| Expected documents **missing** | **1** |

The arithmetic is uncomfortable and it is the central finding of this phase: the corpus appears complete and is not. One file occupies the slot of a document it does not contain, and the document it purports to be is absent.

## 2. Missing documents

### Food Safety and Standards (Ayurveda Aahara) Regulations, 2022 — `MISSING`

The Regulations are not in the corpus, and Phase 3 did not obtain them from FSSAI.

What the corpus actually holds under that filename is the **FSSAI Order of 25 July 2025**, plus a 170-page scanned index of Ayurveda Aahara preparations and their textual references. That file is now verified: all 172 pages match the FSSAI-hosted PDF exactly. It is a genuine, authoritative FSSAI document — of the wrong document.

Three acquisition routes were tried and all failed:

* the FSSAI gazette-notification URL for the 2022 Regulations returns the site's JavaScript shell (3,221 bytes of HTML), not a PDF;
* the FSSAI "Ayurveda Aahara Compendium" downloads successfully — and turns out to be **another recipe index, 647 pages of it**, containing no regulatory prose;
* the FSSAI regulations and notifications listing pages are client-rendered and yield nothing to a non-browser client.

What *is* established about the target: notification **F. No. Stds/SP-05/A-1.Y(01)**, dated **5 May 2022**, made under section 92(2)(v) of the Food Safety and Standards Act, 2006; the draft was published 30 June 2021 and gazette copies were released to the public on 5 July 2021. The document runs to Regulations 1–14 and Schedules A–E.

This matters more than a one-of-twenty-one gap suggests: it is the most Ayurveda-specific instrument on the expected list, and IP-SAKTI's Ayurveda–IP premise leans on it.

### A candidate exists and was not accepted

A copy of the Regulations text supplied as `~/Downloads/pdf.pdf` was assessed in full and **rejected as a source of record** — see §7.

## 3. Stale documents

Staleness was tested against primary instruments, not commentary. Where the only evidence was secondary reporting, the finding is recorded as unconfirmed rather than asserted.

| Document | Local consolidation | Current position | Gap |
|---|---|---|---|
| **Drugs and Cosmetics Rules, 1945** | 31 Dec 2016 | CDSCO serves a 963-page consolidation whose front matter reads *"[As amended vide G.S.R. 360(E) dated 01-07-2024 (w.e.f. 01-07-2024)]"* | **7½ years; 366 more pages** |
| **Drugs and Cosmetics Act, 1940** | 31 Dec 2016 | Amended by the Jan Vishwas Act, 2026 at s. 27A | 9½ years |
| **Patents Rules, 2003** | 23 Jun 2017 | IP India's e-version states *"updated till 15-03-2024"* | 6¾ years |
| **TRIPS** | 1994 unamended text | WTO serves the amended text as default since 23 Jan 2017 | **Article 31bis absent** |
| **Designs Rules, 2001** | Original 2001 text | Amendment Rules 2008, 2014, 2021; IP India publishes a consolidation to 25 Jan 2021 | 3 amendment rounds |
| **GI Rules, 2002** | Original 2002 text | Amendment Rules 2020 and 2025 published by IP India | 2 amendment rounds |
| **Copyright Act, 1957** | Undated | Jan Vishwas Act, 2026 omits s. 67; s. 67 still present | see below |

**Two of these deserve emphasis.**

*TRIPS is the counter-example to "treaties don't go stale."* The corpus copy says so itself, in its own footnote: *"This file reproduces the 1994 unamended version."* It therefore lacks Article 31bis and the associated Annex and Appendix, inserted by the Protocol Amending the TRIPS Agreement, in force since 23 January 2017 — precisely the compulsory-licensing-for-export machinery that a question about Indian pharmaceutical patents or public health would need. A "static treaty" rule would have missed this entirely.

*The Copyright Act is stale against the law, not against its publisher.* The Jan Vishwas (Amendment of Provisions) Act, 2026 (Act 8 of 2026, read directly from the Gazette) omits section 67 at Serial 23 of its Schedule. Section 67 is still present in full in our copy — **and in the Copyright Office's own current copy**. Commencement is by notification under section 1(2) and no commencement notification was located, so the amendment's in-force date is unconfirmed. Re-downloading would not fix this; only a later consolidation will.

**One document was suspected stale and is not.** The Patents Act, 1970 was flagged because the Jan Vishwas Act, 2026 amends it at Serial 35. Reading the local copy settles it: Chapter XX already reads `2[PUNISHMENTS]` and section 119 already reads `[Omitted.]`. The corpus copy incorporates the amendment. Marked current.

## 4. Verified documents — evidence level A (10)

Every page matched a publisher-of-record copy fetched in this phase.

| Document | Publisher matched against | Pages matched |
|---|---|---:|
| `IN-DC-ACT-1940` · `IN-DC-RULES-1945` | cdsco.gov.in | 635 / 635 |
| `IN-FSSAI-AYURVEDA-AAHARA-2022` | fssai.gov.in | 172 / 172 |
| `IN-GI-RULES-2002` | ipindia.gov.in | 71 / 71 (stamp-normalised) |
| `IN-COPYRIGHT-RULES-2013` | copyright.gov.in | 77 / 77 |
| `IN-PPVFR-ACT-2001` | plantauthority.gov.in | 50 / 50 |
| `IN-PPVFR-RULES-2003` | plantauthority.gov.in | 56 / 56 |
| `IN-DESIGNS-RULES-2001` | ipindia.gov.in | 31 / 31 |
| `UN-CBD-1992` | cbd.int | 36 / 36 text; cover image recompressed |
| `UN-NAGOYA-PROTOCOL-2010` | cbd.int | 15 / 15 |

Note that verified and current are different properties: four of these ten are verified **and** superseded.

## 5. Partially verified (level B, 10) and unverified (level C, 2)

**Level B — identity established, copy not matched.** `IN-PATENTS-ACT-1970`, `IN-TM-ACT-1999`, `IN-GI-ACT-1999`, `IN-DESIGNS-ACT-2000`, `IN-PATENTS-RULES-2003`, `IN-TM-RULES-2017`, `IN-BD-RULES-2024` all carry the India Code per-page stamp, which names their publisher beyond doubt. India Code cannot be read by a non-browser client: handle pages return 404, bitstream PDFs 302 to an error page, and `indiacode.gov.in` serves a JavaScript shell. Probed directly on 30 August 2026 — this is a measured fact, not a report. `IN-ABS-REGULATIONS-2025` carries a full Gazette of India Extraordinary masthead but the NBA's own PDF paths are dead (`nbaindia.org` redirects to its homepage; `nbaindia.nic.in` 404s). `WTO-TRIPS` is WTO by its own footnote wording but matched neither WTO PDF on page count. `IN-NATIONAL-IPR-POLICY-2016` has the same 36 pages and the same cover as DPIIT's copy, but DPIIT's is an OCR scan against our native text layer — the same document, two digitisations.

**Level C — provenance unresolved.** `IN-COPYRIGHT-ACT-1957` (57 pages) and `IN-BDA-2002-AMD-2023` (25 pages) carry no imprint of any kind: no India Code stamp, no Gazette masthead, no "As on" line. The Copyright Office's copy of the Act runs to 52 pages, so it is not the same rendering. These two are the only documents in the corpus whose publisher we genuinely cannot name.

## 6. Official source URLs

Recorded per document in `provenance_registry.json` as `official_source_url` and `official_document_url`. Only URLs that were actually fetched, or that appeared verbatim on a fetched official page, are recorded; everything else is `null`. Reachable publishers: `fssai.gov.in`, `cdsco.gov.in`, `plantauthority.gov.in`, `copyright.gov.in`, `ipindia.gov.in`, `cbd.int`, `wto.org`, `egazette.gov.in`, `dpiit.gov.in`. Unreachable to a non-browser client: `indiacode.nic.in` / `indiacode.gov.in`, `nbaindia.nic.in` deep links, `moef.gov.in` deep links.

## 7. Replacement recommendations

Full detail in `source_acquisition_manifest.json`, which separates `existing_source` from `recommended_source` for every document and never conflates them.

**On `~/Downloads/pdf.pdf`.** It was examined in full, and the answer is a qualified no.

It *is* the right text: Regulations 1–14 and Schedules A–D, including the definition of Ayurveda Aahara, the Schedule A list of 71 authoritative books, and the Schedule B category table. But it fails on two counts. **Provenance:** it is an mPDF/Laravel print-to-PDF stamped *"Printed For: 29-08-2026"* and carrying the **LegitQuest** logo on every page — a commercial legal database, which §2 excludes from authoritative sources. **Completeness:** it stops inside Schedule D. Schedule E — the AYURVEDA AAHARA logo that regulation 13(a) makes mandatory on every product label — is referenced but never reproduced, and there is no gazette masthead or signature block.

So it was **not** moved into the corpus, and the existing PDF was not renamed or replaced. It is staged at `data/source_updates/a690fc8c03db07c1-…-legitquest.pdf` and registered as a level-C candidate, `accepted: false`. It is useful for cross-checking while the official text is sought; it is not something a citation should ever rest on. Say the word and it can be admitted anyway, with its evidence level carried through to every chunk it produces.

**Acquired and staged, ready to supersede on approval:** the CDSCO consolidated Drugs Rules, 1945 as amended to 1 July 2024 (963 pages), and the FSSAI, plantauthority, cbd.int, copyright.gov.in and ipindia copies matched above. **Identified but not yet acquired:** IP India's Patents Rules e-version to 15 March 2024, the GI Amendment Rules 2020 and 2025, the Designs Rules consolidation to 25 January 2021, the Copyright (Amendment) Rules 2021, and the WTO amended TRIPS text.

## 8. Version and consolidation findings

`effective_from`, `effective_to`, `consolidated_as_of`, `retrieved_at` and `source_version_id` are recorded per document. Three findings shape how versioning must be treated.

**Newest is not always applicable.** The Copyright Act case shows both directions at once: a passed amendment not yet reflected in any published consolidation, with no located commencement notification. A "latest PDF wins" rule would have silently adopted whichever consolidation happened to be newest.

**India Code re-stamps its "as on" date.** Our Acts carry as-on dates from May and June 2026 that the repository will move again on its own schedule. A future re-download will therefore produce a byte-different file even with no legal change, which is why `source_version_id` keys on the consolidation date rather than the file hash. This could not be confirmed by fetching two snapshots, because the repository is unreachable; it is recorded as inferred.

**Static and current are different questions.** CBD and the Nagoya Protocol are genuinely static and genuinely current. TRIPS is a static text of which we hold a superseded edition. Age alone predicts neither.

## 9. Provenance risks

1. **A wrong-content file in a correctly-named slot.** The FSSAI filename asserts a document it does not contain. Filename-derived identity is not safe anywhere in this corpus; `content_matches_expected_title` is now recorded per document.
2. **India Code is a single point of failure.** Seven documents — including the Patents Act, the Trade Marks Act and the GI Act — depend on a repository that no longer answers automated requests. Their identity is sound and their re-acquisition is currently blocked.
3. **Two documents have no traceable publisher.** The Copyright Act and the Biological Diversity Act consolidations are credible and anonymous.
4. **A superseded treaty text reads as authoritative.** Nothing in the TRIPS chunks signals that Article 31bis is missing; only the source-level record carries that.
5. **Level C material is retrievable and citable.** Citability turns on the locator, not on provenance — deliberately, since the two are independent — so 191 chunks at level C can currently be cited. The evidence level travels on every chunk so a caller can set its own floor.
6. **Amendment instruments are not in the corpus at all.** Where a base text is verified but superseded, the amending instrument is simply absent rather than wrongly represented.

## 10. Recommended source-acquisition order

1. **Food Safety and Standards (Ayurveda Aahara) Regulations, 2022** — missing, most domain-relevant, and not obtainable without a browser session against the FSSAI portal or a direct request to FSSAI.
2. **Drugs and Cosmetics Rules, 1945** to 1 July 2024 — already downloaded and staged; the largest single content gain in the corpus at 366 pages.
3. **TRIPS amended text** including Article 31bis, its Annex and Appendix — and establish whether India has accepted the 2005 Protocol before treating the amended text as binding on India.
4. **Patents Rules, 2003** to 15 March 2024 — needs a browser session to resolve IP India's PDF link.
5. **GI Rules 2020 and 2025 amendments; Designs Rules consolidation to 25 January 2021** — URLs known, straightforward.
6. **Copyright Act, 1957 and Biological Diversity Act, 2002** — acquire publisher copies to resolve the two level-C provenance gaps.
7. **The seven India Code documents** — re-acquire through a JavaScript-capable session; not urgent, since identity is established and six of the seven are current.
8. **Copyright (Amendment) Rules, 2021** — and settle the G.S.R. 225(E) / 226(E) discrepancy against the Gazette rather than against commentary.

---

## Citability invariant (§9)

Implemented and enforced. Every chunk now carries `retrievable`, `citable` and `citability_reason`, and `validate.py` treats a violation as fatal in both directions: a chunk with `locator_unavailable` may never be `citable`, and a located chunk may never be silently withheld. Current state (chunking 0.2.0, after the Phase 5 locator repairs): **3,073 chunks, all retrievable; 3,016 citable; 57 not citable**, all 57 for `locator_unavailable` — 28 of them the FSSAI recipe index, whose locators were withdrawn rather than fabricated, and 7 treaty front-matter chunks (CBD 3, Nagoya 4) that carry no Article at all. The count rose from 52 because Phase 5 withdrew locators that were wrong rather than missing; see `docs/RETRIEVAL_ROUTING.md` §1 and `RETRIEVAL_BENCHMARK.md` §9.1.

Source evidence level is deliberately *not* wired into citability. A correctly located Section 3 of the Patents Act is citable even while its PDF's provenance is unresolved; conversely a level-A document with an unestablished locator is not. Chunks by evidence level: **A 1,685 · B 1,155 · C 191**.

## Verification tooling

`scripts/provenance/verify_source.py` re-runs any comparison in this report:

```bash
.venv/bin/python scripts/provenance/verify_source.py \
  --document-id IN-PPVFR-RULES-2003 \
  --url https://plantauthority.gov.in/sites/default/files/ppvfrarules2003.pdf
.venv/bin/python scripts/provenance/build_registry.py   # regenerate registry + manifest
```

---

## Status

Twenty of twenty-two logical documents have a resolved source identity; ten are matched page-for-page against a publisher of record. Two — the Copyright Act, 1957 and the Biological Diversity Act, 2002 — do not, and their publisher cannot be named from any evidence available. One critical source, the Food Safety and Standards (Ayurveda Aahara) Regulations, 2022, is missing and is documented as missing rather than papered over with a commercial-database substitute.

Because two source identities remain unresolved, this is not a PASS.

PROVENANCE_STATUS: PARTIAL
