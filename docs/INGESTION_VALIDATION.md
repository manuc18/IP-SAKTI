# IP-SAKTI — Phase 2 ingestion validation

Documents processed: **22** logical documents from 21 source PDFs.

## Totals

| Measure | Count |
| --- | ---: |
| Pages processed | 1782 |
| Pages extracted natively | 1565 |
| Pages OCR'd | 217 |
| Structural units parsed | 3418 |
| Chunks generated | 3073 |
| Chunks rejected | 756 |
| Chunks missing a locator | 57 |
| Chunks containing tables | 5 |
| Chunks carrying amendment notes | 1039 |
| Chunks derived from OCR | 162 |
| Hindi lines excluded from retrieval text | 4847 |
| Duplicate chunk groups | 2 |
| Chunks in a duplicate group | 4 |
| Chunks with provenance issues | 0 |

## Validation failures

None. Every fatal check passed:

* every chunk carries `document_id`, `source_sha256` and at least one physical page;
* every source PDF still hashes to the digest recorded in the Phase 1 audit;
* no empty chunks, no `(cid:N)` artefacts, no U+FFFD;
* no Devanagari and no known corrupt-Hindi marker in any retrieval chunk;
* every chunk resolves chunk → page → source PDF;
* jurisdiction, domain and document type are all in the controlled vocabulary.

## OCR

| Document | Pages | Language | Mean confidence | Pages with no recognised text |
| --- | ---: | --- | ---: | ---: |
| `IN-FSSAI-AYURVEDA-AAHARA-2022` | 170 | eng+hin | 89.6 | 42 |
| `IN-PATENTS-RULES-2003` | 47 | eng | 81.94 | 0 |

Rendered at 300 DPI. OCR output is stored in `*.ocr.jsonl` beside the native extraction and never overwrites it; the native record survives under `native_extraction` in the cleaned pages.

## Withdrawn locators

A provision number that the evidence does not support is withdrawn rather than published: a fabricated citation is worse than an honest `locator_unavailable`. Three signals withdraw one — OCR-derived text with no corroborating marginal heading, tabular text with no heading, and text whose function-word density falls below the statutory floor (an index or a list of names, not prose).

| Document | Units withdrawn |
| --- | ---: |
| `IN-DC-RULES-1945` | 4 |
| `IN-FSSAI-AYURVEDA-AAHARA-2022` | 11 |
| `IN-PATENTS-RULES-2003` | 88 |

## Advisory findings

| Check | Count |
| --- | ---: |
| boilerplate_residue | 188 |
| toc_contamination | 4 |

## Per document

| Document | Pages | OCR pages | Units | Chunks | Rejected | No locator | Tables | Amendment notes | Hindi lines dropped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `IN-DC-ACT-1940` | 38 | 0 | 92 | 94 | 3 | 1 | 0 | 91 | 0 |
| `IN-DC-RULES-1945` | 597 | 0 | 1198 | 920 | 558 | 1 | 1 | 395 | 0 |
| `IN-BD-RULES-2024` | 86 | 0 | 48 | 61 | 1 | 1 | 0 | 0 | 1647 |
| `UN-CBD-1992` | 36 | 0 | 68 | 69 | 1 | 3 | 0 | 0 | 0 |
| `IN-COPYRIGHT-RULES-2013` | 77 | 0 | 143 | 133 | 20 | 1 | 1 | 8 | 0 |
| `IN-DESIGNS-RULES-2001` | 31 | 0 | 89 | 94 | 0 | 1 | 0 | 3 | 0 |
| `IN-DESIGNS-ACT-2000` | 18 | 0 | 59 | 50 | 11 | 1 | 0 | 8 | 0 |
| `IN-FSSAI-AYURVEDA-AAHARA-2022` | 172 | 170 | 13 | 29 | 0 | 28 | 3 | 0 | 381 |
| `IN-ABS-REGULATIONS-2025` | 28 | 0 | 23 | 25 | 1 | 1 | 0 | 0 | 549 |
| `IN-GI-RULES-2002` | 71 | 0 | 139 | 142 | 8 | 1 | 0 | 0 | 0 |
| `IN-PATENTS-ACT-1970` | 72 | 0 | 225 | 197 | 31 | 1 | 0 | 176 | 0 |
| `IN-NATIONAL-IPR-POLICY-2016` | 36 | 0 | 9 | 19 | 0 | 4 | 0 | 0 | 0 |
| `IN-BDA-2002-AMD-2023` | 25 | 0 | 86 | 77 | 10 | 1 | 0 | 77 | 0 |
| `IN-COPYRIGHT-ACT-1957` | 57 | 0 | 119 | 114 | 13 | 1 | 0 | 113 | 0 |
| `WTO-TRIPS` | 31 | 0 | 94 | 74 | 20 | 1 | 0 | 0 | 0 |
| `IN-GI-ACT-1999` | 31 | 0 | 99 | 91 | 9 | 1 | 0 | 60 | 0 |
| `IN-PATENTS-RULES-2003` | 102 | 47 | 302 | 279 | 25 | 1 | 0 | 4 | 0 |
| `IN-TM-ACT-1999` | 56 | 0 | 185 | 179 | 12 | 1 | 0 | 101 | 0 |
| `IN-TM-RULES-2017` | 97 | 0 | 186 | 180 | 15 | 1 | 0 | 0 | 2270 |
| `UN-NAGOYA-PROTOCOL-2010` | 15 | 0 | 38 | 41 | 0 | 4 | 0 | 0 | 0 |
| `IN-PPVFR-ACT-2001` | 50 | 0 | 109 | 102 | 11 | 1 | 0 | 0 | 0 |
| `IN-PPVFR-RULES-2003` | 56 | 0 | 94 | 103 | 7 | 1 | 0 | 3 | 0 |

## Duplicate chunks

Reported, not deleted. Identical normalised text most often means a provision reprinted in a Schedule, or an omitted-section marker that reads identically wherever it appears.

* `IN-DC-RULES-1945::part-xv-a-schedule-form-25a-item-1`, `IN-DC-RULES-1945::part-xv-a-schedule-form-28a-item-1`
* `IN-DC-RULES-1945::part-xv-a-schedule-form-25a-item-3`, `IN-DC-RULES-1945::part-xv-a-schedule-form-28-item-3`

## Unresolved extraction issues

### The Ayurveda Aahara Regulations are not in the Ayurveda Aahara PDF

OCR of all 170 scanned pages (95,882 characters at 89.6% mean confidence) returns no regulatory prose whatsoever: zero occurrences of *short title*, *commencement*, *definitions*, *these regulations*, *Schedule A/B* or *licence*, and one occurrence of *shall* in the whole enclosure. Page 3 opens with the word *Index*; page 172 closes with a list of Ayurvedic reference works. The scanned enclosure is an **index of Ayurveda Aahara preparations and their authoritative textual references**, annexed to the 2-page FSSAI covering Order of 25 July 2025.

This corrects the Phase 1 audit, which inferred from the covering Order that the Regulations text and the Schedule A/B lists were behind the scan. They are not. **The Food Safety and Standards (Ayurveda Aahara) Regulations, 2022 are absent from the corpus** and must be sourced separately — which matters, because they are the most Ayurveda-specific instrument on the expected list. All 28 chunks from this document are correctly marked `locator_unavailable`.

### Hindi is excluded from retrieval, not repaired

4847 Devanagari-bearing lines were dropped from the retrieval text across the five bilingual documents. Phase 1 established that every one of them has a defective ToUnicode CMap: the output is well-formed Unicode in the correct block and the wrong characters, so no encoding check catches it. A line is dropped if it contains *any* Devanagari, because OCR of a bilingual scan interleaves the scripts within a line. Nothing is transliterated, translated or corrected — `raw_text` retains every dropped line verbatim for Phase 3.

### Two consolidations are stale

The Patents Rules, 2003 are consolidated only to 23 June 2017 and the Drugs and Cosmetics Rules, 1945 only to 31 December 2016. Their chunks are faithful to the PDFs and the PDFs are out of date.

### Provenance remains unverified

Every chunk carries `provenance_status: PROVENANCE_UNVERIFIED`, inherited from the Phase 1 audit: 15 of the 21 source PDFs are browser re-encodings whose publisher metadata was destroyed, and TRIPS carries no publisher imprint at all. Ingestion cannot fix this; only re-fetching each document from its publisher of record can.

### Layout artefacts that survive extraction

The Drugs and Cosmetics compendium sets marginal notes in a side column, which the text layer interleaves into the body — clauses occasionally appear out of reading order within a provision. This is a property of the PDF, not of the pipeline, and is left uncorrected rather than guessed at.

VALIDATION_STATUS: PASS
