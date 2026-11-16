# IP-SAKTI — Corpus Audit (v0.1.0)

**Scope:** read-only technical audit of `~/Downloads/IP-SAKTI Corpus`.
**Date:** 2026-08-29T17:08:39+00:00
**Tooling:** `pypdf 6.16.2` in a project-local virtualenv. Text-layer extraction only — **no OCR was run, and no file in the corpus was modified, renamed, moved or rewritten.**
**Companion artefact:** `data/metadata/corpus_manifest.json` (one record per source document).

---

## 1. Executive summary

The corpus contains **21 PDFs / 1,782 pages / 47.4 MB**, from which **4,368,249 characters** of text were recovered without OCR. All **20** documents of the Core Legal Corpus V0.1 are present, plus one unlisted bonus document. Nothing is encrypted, truncated or corrupt at the container level; every file carries a valid `%PDF` header and `%%EOF` marker and opened without error.

**The corpus is in good shape for an English-language ingestion pipeline, and in poor shape for a Hindi one.** Three findings drive the Phase 2 plan:

1. **One document is effectively unusable as delivered.** The *Food Safety and Standards (Ayurveda Aahara) Regulations, 2022* file is 170 of 172 pages of pure page images. The only extractable text is a two-page FSSAI covering Order dated 25 July 2025 — the Regulations themselves, and the Schedule A/B lists that matter most for Ayurveda, exist only as pictures. This is also the single most domain-relevant document in the corpus.
2. **The Patents Rules, 2003 lose their Forms.** Pages 1–55 extract cleanly; pages 56–102 — the entire body of statutory Forms and fee Schedules — are page images. Separately, this copy is consolidated only to **23 June 2017** and therefore predates several later amendment rounds.
3. **Hindi text extraction is unreliable in every bilingual document.** The Devanagari is not missing; it is *silently wrong* — valid Unicode produced by defective font-to-Unicode mappings, so it passes every automated encoding check while being unreadable. See §6. For a project whose premise is multilingual citation-grounded answering, this is the highest-risk finding, because a naive pipeline would embed and cite this text without any error surfacing.

No document's provenance can be verified from the files alone. Every record is marked `PROVENANCE_UNVERIFIED` (§4).

---

## 2. File inventory

All 21 files sit flat in the corpus root; there are no subdirectories and no non-PDF files.

| # | Filename | Size | Pages | Chars extracted | Chars/page | Text-native | OCR needed | Lang |
|---|---|---:|---:|---:|---:|:-:|:-:|:-:|
| 1 | `IN-PAT-ACT-1970.pdf` | 0.7 MB | 72 | 251,880 | 3,498 | yes | no | en |
| 2 | `The-Patents-Rules-2003-Updated-till-23-June-2017.pdf` | 3.5 MB | 102 | 128,726 | 1,262 | **no** | **yes** | en |
| 3 | `Trade Marks Act, 1999.pdf` | 0.7 MB | 56 | 199,036 | 3,554 | yes | no | en |
| 4 | `Trademarks rules 2017.pdf` | 1.9 MB | 97 | 344,174 | 3,548 | yes | no | en+hi |
| 5 | `The Geographical Indications of Goods (Registration and Protection) Act, 1999 Act No. 48 of 1999.pdf` | 0.4 MB | 31 | 106,559 | 3,437 | yes | no | en |
| 6 | `Geographical Indications.pdf` | 0.2 MB | 71 | 184,314 | 2,596 | yes | no | en |
| 7 | `Designs Act, 2000.pdf` | 0.3 MB | 18 | 58,695 | 3,261 | yes | no | en |
| 8 | `Design Rules, 2001.pdf` | 0.1 MB | 31 | 66,431 | 2,143 | yes | no | en |
| 9 | `THE COPYRIGHT ACT, 1957.pdf` | 0.8 MB | 57 | 159,217 | 2,793 | yes | no | en |
| 10 | `Copyright Rules 2013.pdf` | 0.5 MB | 77 | 153,231 | 1,990 | yes | no | en |
| 11 | `THE BIOLOGICAL DIVERSITY ACT, 2002.pdf` | 0.3 MB | 25 | 88,745 | 3,550 | yes | no | en |
| 12 | `Biological Diversity Rules, 2024.pdf` | 2.4 MB | 86 | 201,536 | 2,343 | yes | no | en+hi |
| 13 | `ppvfract2001.pdf` | 0.2 MB | 50 | 123,329 | 2,467 | yes | no | en |
| 14 | `ppvfrarules2003.pdf` | 0.2 MB | 56 | 144,425 | 2,579 | yes | no | en |
| 15 | `2016DrugsandCosmeticsAct1940Rules1945.pdf` | 7.2 MB | 635 | 1,783,661 | 2,809 | yes | no | en |
| 16 | `Food Safety and Standards (Ayurveda Aahara) Regulations, 2022.pdf` | 9.4 MB | 172 | 3,588 | 21 | **no** | **yes** | en+hi |
| 17 | `TRIPS Agreement.pdf` | 0.2 MB | 31 | 87,573 | 2,825 | yes | no | en |
| 18 | `Convention on Biological Diversity (CBD).pdf` | 4.3 MB | 36 | 67,026 | 1,862 | yes | no | en |
| 19 | `nagoya-protocol-en.pdf` | 0.4 MB | 15 | 56,770 | 3,785 | yes | no | en |
| 20 | `NIPRP.pdf` | 12.1 MB | 36 | 75,632 | 2,101 | yes | no | en |
| 21 | `GNABSREG_2025.pdf` | 1.5 MB | 28 | 83,701 | 2,989 | yes | no | en+hi |

**Totals:** 21 files · 1,782 pages · 47.4 MB · 4,368,249 characters.

Structural facts common to the whole corpus: no file is encrypted; no file failed to open; no page raised an extraction exception; no two files share a SHA-256; and no duplicate page content was detected within any file.

### File-type breakdown

| Class | Count |
|---|---:|
| PDF | 21 |
| Text / image / other | 0 |
| Text-native PDFs (usable without OCR) | 19 |
| PDFs requiring OCR | 2 |

---

## 3. Expected-document coverage

Each file was confirmed by reading its front matter, not by trusting its filename — which mattered, because two filenames are misleading.

| # | Expected document | Status | File | Evidence used |
|---|---|---|---|---|
| 1 | Patents Act, 1970 | **FOUND** | `IN-PAT-ACT-1970.pdf` | Title page: “The Patents Act, 1970 (ACT NO. 39 OF 1970) [As on the 5th May, 2026]”. |
| 2 | Patents Rules, 2003 | **FOUND** | `The-Patents-Rules-2003-Updated-till-23-June-2017.pdf` | Title page: “THE PATENTS RULES, 2003 … updated till 23-06-2017”. |
| 3 | Trade Marks Act, 1999 | **FOUND** | `Trade Marks Act, 1999.pdf` | Title page: “The Trade Marks Act, 1999 (ACT NO. 47 OF 1999) [As on the 1st June, 2026]”. |
| 4 | Trade Marks Rules, 2017 | **FOUND** | `Trademarks rules 2017.pdf` | Gazette header, Part II Sec. 3(i); notification G.S.R. 199(E). |
| 5 | Geographical Indications of Goods (Registration and Protection) Act, 1999 | **FOUND** | `The Geographical Indications of Goods (Registration and Protection) Act, 1999 Act No. 48 of 1999.pdf` | Title page: “Act No. 48 of 1999 [As on the 1st June, 2026]”. |
| 6 | Geographical Indications of Goods (Registration and Protection) Rules, 2002 | **FOUND** | `Geographical Indications.pdf` | **Filename says only “Geographical Indications”.** Page 1 reads: “Text of The Geographical Indications of Goods (Registration and Protection) **Rules, 2002** … Gazette … Dated 8th March, 2002”, G.S.R. 176(E). Confirmed as the Rules, not a second copy of the Act. |
| 7 | Designs Act, 2000 | **FOUND** | `Designs Act, 2000.pdf` | Title page: “The Designs Act, 2000 (ACT NO. 16 OF 2000) [As on the 15th June, 2026]”. |
| 8 | Designs Rules, 2001 | **FOUND** | `Design Rules, 2001.pdf` | Notification of 11 May 2001, S.O. 414(E), enacting “THE DESIGNS RULES, 2001”. |
| 9 | Copyright Act, 1957 | **FOUND** | `THE COPYRIGHT ACT, 1957.pdf` | Title page: “THE COPYRIGHT ACT, 1957 (14 OF 1957)”. |
| 10 | Copyright Rules, 2013 | **FOUND** | `Copyright Rules 2013.pdf` | Notification G.S.R. 172(E) of 14 March 2013; a G.S.R. 788(E) amendment of 10 August 2016 is referenced in-text. |
| 11 | Biological Diversity Act, 2002 (as amended by the 2023 Amendment) | **FOUND** | `THE BIOLOGICAL DIVERSITY ACT, 2002.pdf` | **Amendment coverage verified.** Marginal notes throughout read “Subs./Ins. by Act 10 of 2023 … (w.e.f. 1-4-2024)”, and s. 58 is shown as “Omitted by the Biological Diversity (Amendment) Act, 2023 (10 of 2023)”. Sections 10A, 36A, 36B, 52A, 55A, 55B and 59A are present. This **is** the 2023-amended consolidated text. |
| 12 | Biological Diversity Rules, 2024 | **FOUND** | `Biological Diversity Rules, 2024.pdf` | Gazette of India Extraordinary, Part II Sec. 3(i); G.S.R. 665(E). |
| 13 | Protection of Plant Varieties and Farmers' Rights Act, 2001 | **FOUND** | `ppvfract2001.pdf` | Running header “THE GAZETTE OF INDIA EXTRAORDINARY”; notification of 30 October 2001. |
| 14 | Protection of Plant Varieties and Farmers' Rights Rules, 2003 | **FOUND** | `ppvfrarules2003.pdf` | Notification of 12 September 2003; heavy Rule / sub-rule / Schedule structure. |
| 15 | Drugs and Cosmetics Rules, 1945 | **FOUND** | `2016DrugsandCosmeticsAct1940Rules1945.pdf` | **Filename is a compendium name.** Title page: “GOVERNMENT OF INDIA / MINISTRY OF HEALTH AND FAMILY WELFARE (Department of Health) — THE DRUGS AND COSMETICS ACT, 1940 (23 OF 1940) … and THE DRUGS AND COSMETICS RULES, 1945 (As amended up to the 31st December, 2016)”. The Rules occupy pages 31–546 of the printed pagination. |
| 16 | Food Safety and Standards (Ayurveda Aahara) Regulations, 2022 | **FOUND** | `Food Safety and Standards (Ayurveda Aahara) Regulations, 2022.pdf` | **Content differs from the filename.** Pages 1–2 are an FSSAI Order of 25 July 2025 about Ayurveda Aahara under the 2022 Regulations, ref. SS-T0SP05(EC)/2/2022-Standard-FSSAI I/29912/2025. The Regulations themselves are the scanned enclosure on pages 3–172 and could not be read without OCR. |
| 17 | Agreement on Trade-Related Aspects of Intellectual Property Rights (TRIPS) | **FOUND** | `TRIPS Agreement.pdf` | Page 1: “ANNEX 1C — AGREEMENT ON TRADE-RELATED ASPECTS OF INTELLECTUAL PROPERTY RIGHTS”; 120 Article headings detected. |
| 18 | Convention on Biological Diversity (CBD), 1992 | **FOUND** | `Convention on Biological Diversity (CBD).pdf` | Cover: “Convention on Biological Diversity — text and annexes”; `www.cbd.int` appears in the running text; date 22 May 1992 in front matter. |
| 19 | Nagoya Protocol on Access to Genetic Resources and Benefit-Sharing | **FOUND** | `nagoya-protocol-en.pdf` | Full protocol title present; `www.cbd.int`; dates 5 June 1992 / 29 December 1993 / 29 October 2010 in front matter. |
| 20 | National Intellectual Property Rights (IPR) Policy, 2016 | **FOUND** | `NIPRP.pdf` | Cover: “Creative India; Innovative India — Government of India, Ministry of Commerce and Industry, DPIIT”. In-text messages state “the adoption of the National IPR Policy back in May 2016”. The InDesign job stamp `national ipr policy__AA.indd … 03/02/20` shows this is a **2020 print edition** of the 2016 policy. |

### Documents present but not on the expected list

| Document | File | Notes |
|---|---|---|
| Biological Diversity (Access to Biological Resources and Associated Knowledge and Fair and Equitable Sharing of Benefits) Regulations, 2025 | `GNABSREG_2025.pdf` | National Biodiversity Authority notification, Chennai, 29 April 2025; Gazette of India Extraordinary, Part III Sec. 4, No. 332, 30 April 2025; F. No. NBA/Tech/EC/9/14/32. It expressly supersedes the 2014 ABS Guidelines and operates under the Biological Diversity Rules, 2024. **Directly on-topic for ABS and should be adopted into the corpus** as item 21. |

**Coverage: 20 of 20 expected documents FOUND. 0 POSSIBLY FOUND. 0 MISSING.**

---

## 4. Provenance findings

**Every document is marked `PROVENANCE_UNVERIFIED`, without exception.** Nothing in the corpus establishes an official chain of custody: there are no download receipts, no capture URLs, no checksums from a publisher, and no signatures. What follows is the internal evidence only.

### 4.1 The re-encoding problem

Fifteen of the 21 files report `/Producer: PDFium` with a `/CreationDate` of **29 August 2026, between 20:35 and 21:45** — matching the files' own mtimes and therefore the collection session, not the documents. PDFium is the Chrome PDF engine, so these files were re-generated by the browser on download or print-to-PDF. **Original publisher metadata was destroyed in that step**, which is precisely the metadata provenance verification would rely on.

Only six files retain pre-collection metadata:

| File | Producer | Original creation date | Author field |
|---|---|---|---|
| `The-Patents-Rules-2003-Updated-till-23-June-2017.pdf` | Adobe Acrobat 10.1.16 | 2017-05-26 (mod 2017-06-23) | User |
| `Trademarks rules 2017.pdf` | PDFCreator 2.3.0.103 | 2017-03-09 (mod 2017-03-10) | aaa (title: `1256gi`) |
| `IN-PAT-ACT-1970.pdf` | Microsoft Word 2021 | 2026-06-11 | Naseem |
| `Trade Marks Act, 1999.pdf` | Microsoft Word 2021 | 2026-06-03 | deo ld |
| `The Geographical Indications … Act, 1999 ….pdf` | Microsoft Word 2021 | 2026-06-03 | deo ld |

The three Word-produced files carry personal author names, which is consistent with the India Code publication workflow but is not proof of it.

### 4.2 Internal provenance evidence, by document

| Evidence found in document | Documents |
|---|---|
| **India Code watermark** in the page footer (indiacode.nic.in, the Government of India legislative repository) | Patents Act 1970 · Patents Rules 2003 · Trade Marks Act 1999 · Trade Marks Rules 2017 · GI Act 1999 · GI Rules 2002 · Designs Act 2000 · Biological Diversity Rules 2024 |
| **Gazette of India, Extraordinary** running header with Part/Section citation | Biological Diversity Rules 2024 (Part II Sec. 3(i)) · ABS Regulations 2025 (Part III Sec. 4) · Trade Marks Rules 2017 (Part II Sec. 3(i)) · PPVFR Act 2001 |
| **Ministry letterhead on the title page** | Drugs & Cosmetics compendium (MoHFW, Dept. of Health) · Ayurveda Aahara (FSSAI, Science and Standards Division) · National IPR Policy (DPIIT) |
| **Publisher domain visible in text** | `www.cbd.int` (CBD, Nagoya) · `http://www.ipindia.gov.in` (Patents Rules) · `copyright.gov.in` (Copyright Rules) · `fssai.gov.in` (Ayurveda Aahara) |
| **Statutory instrument number recovered** | G.S.R. 176(E), 199(E), 172(E), 665(E) · S.O. 414(E) · Act Nos. 39/1970, 47/1999, 48/1999, 16/2000, 14/1957, 18/2003, 23/1940 · F. No. NBA/Tech/EC/9/14/32 |
| **No publisher imprint of any kind** | TRIPS Agreement — the file is a bare “ANNEX 1C” text with no WTO branding, masthead, URL or document symbol. Attribution to the WTO is an inference from the Annex framing alone. |

### 4.3 Currency and version concerns

| Document | Stated consolidation point | Concern |
|---|---|---|
| Patents Rules, 2003 | “updated till 23-06-2017” | **Stale.** Later amendment rounds are not in this text. A user asking a patent-procedure question would receive a 2017-era answer presented with full confidence. |
| Drugs and Cosmetics Rules, 1945 | “As amended up to the 31st December, 2016” | **Stale by ~9 years**, and this is a live regulatory area for Ayurvedic drug licensing. |
| Patents Act 1970 / Trade Marks Act 1999 / GI Act 1999 / Designs Act 2000 | “As on” 5 May / 1 June / 1 June / 15 June 2026 | Current, and each states its own as-on date — usable as a version string. |
| Biological Diversity Act, 2002 | Incorporates Act 10 of 2023 w.e.f. 1-4-2024 | Current. |
| National IPR Policy, 2016 | 2020 print edition | Content is the 2016 policy; the file is a later reprint with added ministerial forewords. Cite the policy, not the print run. |
| Copyright Act 1957 · Copyright Rules 2013 · Designs Rules 2001 · GI Rules 2002 · PPVFR Act/Rules · CBD · Nagoya · TRIPS | No as-on statement | Version cannot be established from the document. Leave `version: null` rather than assume the text is current. |

---

## 5. OCR requirements

**2 documents require OCR, covering 217 pages (12.2% of the corpus).**

| Document | OCR pages | Of total | Priority | What is lost without OCR |
|---|---:|---:|---|---|
| Food Safety and Standards (Ayurveda Aahara) Regulations, 2022 | **170** | 172 | **CRITICAL** | The entire Regulations text and the Schedule A/B authoritative-text lists. Only a 2-page covering Order survives. This is the corpus's most Ayurveda-specific document. |
| Patents Rules, 2003 | **47** | 102 | **HIGH** | Pages 56–102: all statutory Forms and fee Schedules. The text layer on these pages contains only the `Page N of 66` footer and the India Code watermark. |

### Pages that look like OCR candidates but are not

The naive heuristic “page has little text ⇒ page needs OCR” flags 7 more pages. Each was inspected individually and dismissed:

| Document | Pages | Actual content |
|---|---|---|
| National IPR Policy, 2016 | 1, 4, 6, 8 | Vector cover art and full-page portrait plates facing the message text on pages 5, 7, 9. Pages 2, 10, 12, 32–35 are blanks, `NOTES` pages and cover credits. No policy text is lost. |
| Convention on Biological Diversity | 1 | Cover artwork. Pages 2, 35, 36 are blanks and the printer's colophon. |
| Copyright Rules, 2013 | 1 | Image-only cover sheet; the 76 substantive pages are text-native. |
| Copyright Act, 1957 | 1 | Title page bearing an emblem image alongside extractable title text. |

**Recommended OCR configuration** when Phase 2 reaches it: Tesseract with `eng` for the Patents Rules forms, and `eng+hin` for the Ayurveda Aahara scans (that document is bilingual). Render at ≥300 DPI. Both are page-image sources, so OCR output should be written to a **separate derived-text store** and flagged `text_source: "ocr"` per chunk, never silently merged with publisher text — OCR'd statutory text carries a materially different error profile and citations drawn from it need to say so.

---

## 6. Text extraction quality

### 6.1 English — clean across the corpus

English extraction succeeded everywhere it was attempted. Diagnostics for control characters, `(cid:N)` artefacts, U+FFFD replacement characters and vowel-less token runs came back essentially at zero for all 21 files (highest non-printable ratio outside the scanned FSSAI file: 0.026%). Word spacing, hyphenation and section numbering all survive. Extraction density of 1,900–3,600 characters per page is consistent with clean two-column and single-column statutory typesetting.

### 6.2 Hindi — corrupt in every bilingual document

This is the audit's most consequential finding, and it is **invisible to standard encoding checks**. The Devanagari that comes out of these files is well-formed Unicode in the correct code block. It is simply the *wrong characters*, because the PDFs embed subset fonts with defective or absent `ToUnicode` CMaps. Nothing throws an error; nothing produces a replacement character; a language detector still says “Hindi”.

Detection method: count occurrences of Hindi legal vocabulary that **must** appear in any Gazette text, against the specific mis-mappings observed.

| Document | Correct forms found | Corrupted forms found | Verdict |
|---|---|---|---|
| Biological Diversity Rules, 2024 | `राजपत्र` ×0 | `रािपत्र` ×44, `प्राजधकरण` ×166, `सिस्ट्र्` ×40 | **Systematically corrupt.** Consistent substitution pattern (ध→जध, य→र्, द→ि). |
| ABS Regulations, 2025 | `राजपत्र` ×14 | `अजधजनयम` ×50, `जवजनयम` ×38, `जनयम` ×105 | **Partially corrupt** — the file mixes correctly and incorrectly mapped font programs, so quality varies page to page. |
| Trade Marks Rules, 2017 | — | **13,888 Private-Use-Area codepoints**; vowel-sign reordering (`िनमाᭅण` for `निर्माण`); running header in legacy Chanakya 8-bit ASCII (`Hkkjr dk jkti=k`) | **Unusable.** |
| Ayurveda Aahara Regulations, 2022 | — | Matras and conjuncts dropped: `खा सुरा` for `खाद्य सुरक्षा` | **Unusable** (and only 2 pages extract at all). |
| National IPR Policy, 2016 | — | Cover tagline in legacy Chanakya: `jpukRed Hkkjr;` for `रचनात्मक भारत` | Cover only; the document is otherwise English. |

**Consequence for IP-SAKTI.** Every bilingual document in this corpus is an English/Hindi Gazette pair in which **the English half is trustworthy and the Hindi half is not**. Ingesting the Hindi as-is would poison Hindi retrieval and, worse, produce confidently-cited Hindi answers built on text that was never in the Gazette. The safe position for v0.1 is to **ingest the English column only and drop the Devanagari at chunk time**, then reconstruct Hindi separately via OCR later if multilingual answering is required. Hindi-language support should be treated as an open Phase 3 problem, not as something this corpus already provides.

### 6.3 Other quality signals

- **Duplicate pages:** none. Normalised page-text hashing across all 1,782 pages found zero repeats within any document.
- **Corruption:** none. All 21 files have valid headers and `%%EOF` markers; zero extraction exceptions across 1,782 pages.
- **Blank pages:** 11 genuinely blank pages, all in NIPRP (7) and CBD (3), plus 1 in the Copyright Rules cover. They should be dropped at ingestion.
- **Encoding:** no UTF-8 decode failures, no mojibake in Latin text.

---

## 7. Structural parsing observations

Legal-structure markers were counted per document. Every Indian statute and rule-set in the corpus carries dense, machine-detectable structure — which is exactly what a citation-grounded system needs.

| Document | Chapter | Section | Sub-sec | Clause | Rule | Sub-rule | Schedule | Annexure | Article | Regulation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IN-PATENTS-ACT-1970 | 51 | 747 | 576 | 392 | 0 | 0 | 5 | 0 | 0 | 0 |
| IN-PATENTS-RULES-2003 | 33 | 277 | 446 | 132 | 69 | 31 | 40 | 3 | 7 | 0 |
| IN-TM-ACT-1999 | 32 | 505 | 424 | 387 | 0 | 0 | 8 | 0 | 0 | 0 |
| IN-TM-RULES-2017 | 9 | 339 | 807 | 225 | 91 | 23 | 42 | 0 | 1 | 0 |
| IN-GI-ACT-1999 | 22 | 262 | 231 | 186 | 0 | 0 | 3 | 0 | 0 | 0 |
| IN-GI-RULES-2002 | 8 | 342 | 425 | 50 | 157 | 29 | 55 | 1 | 0 | 0 |
| IN-DESIGNS-ACT-2000 | 23 | 146 | 136 | 113 | 0 | 0 | 0 | 0 | 0 | 0 |
| IN-DESIGNS-RULES-2001 | 0 | 149 | 110 | 64 | 28 | 3 | 11 | 0 | 0 | 0 |
| IN-COPYRIGHT-ACT-1957 | 27 | 151 | 235 | 357 | 0 | 0 | 3 | 0 | 0 | 0 |
| IN-COPYRIGHT-RULES-2013 | 19 | 447 | 295 | 421 | 52 | 9 | 17 | 0 | 0 | 0 |
| IN-BDA-2002-AMD-2023 | 23 | 290 | 183 | 144 | 0 | 0 | 2 | 0 | 1 | 0 |
| IN-BD-RULES-2024 | 0 | 166 | 586 | 431 | 24 | 11 | 3 | 1 | 0 | 0 |
| IN-PPVFR-ACT-2001 | 14 | 190 | 352 | 262 | 0 | 0 | 0 | 0 | 1 | 0 |
| IN-PPVFR-RULES-2003 | 9 | 412 | 423 | 96 | 107 | 36 | 75 | 3 | 0 | 0 |
| IN-DC-ACT1940-RULES1945 | 94 | 4162 | 1473 | 3208 | 326 | 56 | 569 | 29 | 0 | 0 |
| IN-FSSAI-AYURVEDA-AAHARA-2022 | 0 | 3 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| WTO-TRIPS | 0 | 153 | 0 | 53 | 0 | 0 | 0 | 3 | 120 | 0 |
| UN-CBD-1992 | 0 | 93 | 0 | 67 | 0 | 0 | 0 | 11 | 82 | 0 |
| UN-NAGOYA-PROTOCOL-2010 | 0 | 79 | 0 | 109 | 0 | 0 | 0 | 7 | 54 | 0 |
| IN-NATIONAL-IPR-POLICY-2016 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| IN-ABS-REGULATIONS-2025 | 0 | 103 | 147 | 94 | 3 | 0 | 1 | 0 | 0 | 10 |

*(Counts are regex hits over the extracted text, so they are indicative volume, not validated citation counts. The scanned FSSAI file scores near zero purely because its text layer is absent.)*

### What this implies for the chunker

- **Two distinct citation grammars are in play.** Indian instruments are addressed as *Chapter → Section → sub-section → clause* (Acts) and *Rule → sub-rule → clause → Schedule* (Rules). The three treaties — TRIPS (120 Articles), CBD (82), Nagoya (54) — use *Article → paragraph* and have **zero** Section/Rule/Schedule structure. A single chunker that assumes Indian statutory numbering will mis-parse all three international instruments. Plan for two parsers from the start.
- **Front-matter “ARRANGEMENT OF SECTIONS” blocks must be excluded.** Most India Code files open with a table of contents listing every section heading. Left in, it becomes a high-similarity duplicate of the real provisions and will dominate retrieval with content-free hits.
- **Amendment footnotes are inline and dense.** The Biological Diversity Act carries footnotes such as “Subs. by Act 10 of 2023, s. 19, for the proviso (w.e.f. 1-4-2024)” at the foot of nearly every page, interleaved into the extracted text stream. These are genuinely valuable — they are what lets the system answer “what changed in 2023?” — but if they are not separated from operative text they will fragment provisions mid-sentence. Extract them into a `amendment_notes` field on the chunk rather than discarding or inlining them.
- **Running headers and footers repeat on most pages** and must be stripped: `Drugs and Cosmetics Rules N` (582 pages), `IndiaCode` (up to 102 pages/file), `THE GAZETTE OF INDIA : EXTRAORDINARY [PART II—SEC. N(i)]`, `Page N of 66`, `Convention on Biological Diversity`, `THE GAZETTE OF INDIA EXTRAORDINARY N`, `National IPR Policy`, and the InDesign job stamp `national ipr policy__AA.indd …` which leaks into NIPRP body text.
- **Tables are present but sparse**, concentrated in the fee and rate schedules: Trade Marks Rules 2017 (34 pages), Drugs & Cosmetics (10), PPVFR Rules 2003 (4), BD Rules 2024 (4), NIPRP (3), ABS Regulations 2025 (2). `pypdf` flattens these into whitespace-separated runs and loses column alignment. Fee schedules are a likely user question type, so these pages warrant a table-aware extractor (`pdfplumber` or `camelot`) rather than the plain text path.
- **Page numbering is inconsistent.** Only about half the files carry a detectable page-number pattern, and the Patents Rules PDF has 102 physical pages against a printed `Page N of 66` sequence. **Cite by statutory locator (section/rule/article), never by PDF page index** — the two do not correspond.

---

## 8. Duplicates and ambiguities

**No duplicates.** All 21 SHA-256 digests are distinct, and no document duplicates another's subject matter. No repeated page content was found within any file.

**Four filename ambiguities**, all resolved by reading the documents:

| File | Ambiguity | Resolution |
|---|---|---|
| `Geographical Indications.pdf` | Name gives no clue whether this is the Act or the Rules, and a second GI file exists | It is the **GI Rules, 2002** (G.S.R. 176(E), 8 March 2002). The Act is the separate, fully-named file. Not a duplicate. |
| `Food Safety and Standards (Ayurveda Aahara) Regulations, 2022.pdf` | Named as the Regulations | Contains a **25 July 2025 FSSAI covering Order**; the Regulations are the scanned enclosure. Both title and date differ from the filename. |
| `2016DrugsandCosmeticsAct1940Rules1945.pdf` | Four years in one name | A single **compendium** holding the Drugs and Cosmetics Act, 1940 *and* the Rules, 1945, both consolidated to 31 December 2016. Expected item 15 is the Rules; the Act comes along as a bonus. Should be split into two logical documents at ingestion. |
| `GNABSREG_2025.pdf` | Opaque abbreviation | The **ABS Regulations, 2025** issued by the National Biodiversity Authority. Not on the expected list. |

`NIPRP.pdf` is likewise opaque but resolves cleanly to the National IPR Policy.

---

## 9. Missing documents

**None of the 20 expected documents is missing.**

The following gaps are *content* gaps rather than missing files, and are worth recording now because they will shape what the system can honestly answer:

1. **The Ayurveda Aahara Regulations are present but unreadable** until OCR runs. Functionally, the corpus's flagship Ayurveda document is unavailable today.
2. **The Patents Rules Forms are present but unreadable** (pages 56–102), and the whole instrument is consolidated only to 2017.
3. **The Drugs and Cosmetics Rules stop at 31 December 2016.**
4. **Hindi content is present but not trustworthy** in all five bilingual documents (§6.2).
5. **No traditional-knowledge source is in the corpus at all.** For a system positioned on Ayurveda and traditional knowledge, there is no TKDL documentation, no Ayush ministry guidance, no NBA guidelines beyond the 2025 ABS Regulations, and no case law. This is not a defect in the delivered corpus — none of it was on the expected list — but it bounds what v0.1 can answer, and the gap should be visible to whoever sets user expectations.

---

## 10. Recommended next ingestion steps

Ordered, and deliberately stopping short of retrieval or generation.

**Phase 2a — normalise and extract (no ML).**
1. **Freeze provenance before anything else.** Copy the corpus into `data/raw/` under content-addressed names, record each SHA-256, and never read from `~/Downloads` again. The corpus root is a user Downloads folder; it is not a stable base for a citation system.
2. **Resolve provenance properly.** For each of the 21 documents, re-fetch from the publisher of record (India Code, IP India, FSSAI, NBA, WTO, CBD Secretariat), record the URL and retrieval timestamp, and diff against the local copy. Only then may `provenance_status` move off `PROVENANCE_UNVERIFIED`. **Until that happens, no answer should be presented to a user as authoritative** — the current files are browser re-encodings with publisher metadata stripped, and that is not a defensible basis for legal citation.
3. **Replace the two stale instruments** — Patents Rules 2003 (2017) and Drugs & Cosmetics Rules 1945 (2016) — with current consolidations, and add the ABS Regulations 2025 as corpus item 21.
4. **Build the English text extraction pass** with a layout-aware extractor, emitting per-page text plus coordinates. Strip the running headers/footers catalogued in §7, drop the `ARRANGEMENT OF SECTIONS` front matter, drop the 11 blank pages, and route the table-heavy pages through a table extractor.
5. **Drop Devanagari at extraction time for v0.1**, recording per document that Hindi was excluded and why. Do not ingest text known to be wrong.

**Phase 2b — OCR the two gaps.** Run Tesseract at ≥300 DPI (`eng+hin` for Ayurveda Aahara, `eng` for the Patents Rules forms) into a separate derived store, tagged `text_source: "ocr"` so downstream citations can disclose it. Spot-check accuracy against a manually transcribed sample before accepting the output.

**Phase 2c — structural parsing.** Build the two parsers §7 calls for — Indian statutory (Chapter/Section/sub-section/clause and Rule/sub-rule/Schedule) and treaty (Article/paragraph) — and emit a citation locator for every chunk. Split the Drugs & Cosmetics compendium into its Act and Rules components. Separate amendment footnotes into their own field.

**Phase 2d — validate before embedding.** Assert that every chunk carries a resolvable locator, that section numbers within each document form a sensible sequence with no unexplained gaps, and that no chunk contains header/footer boilerplate or Devanagari. Fix the extractor until these pass. **Embeddings should not be generated until this gate is green** — a citation-grounded system inherits every extraction error as a wrong citation, and re-embedding after the fact is more expensive than validating first.

Only after 2d does the vector-store and retrieval work in the wider architecture become worth starting.

---

## Appendix — reproducing this audit

```bash
python3 -m venv .venv && ./.venv/bin/pip install pypdf
./.venv/bin/python scripts/audit_corpus.py "~/Downloads/IP-SAKTI Corpus"
```

The audit reads with `open(path, "rb")` and `pypdf.PdfReader` only. No write, rename, move or delete operation touches the corpus directory.

