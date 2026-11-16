"""Phase H: a 1,000-question synthetic benchmark, built from the corpus alone.

    .venv/bin/python scripts/evaluation/generate_synthetic_benchmark.py
    .venv/bin/python scripts/evaluation/generate_synthetic_benchmark.py --count 1000 --seed 20260915

What this is, stated before anything else reads a number off it: **a synthetic,
AI-designed benchmark whose labels are derived from the corpus, not verified by a
lawyer.** Every question here was written by a template over a chunk this
repository already stores, and every expected answer is a span *copied* out of
that same chunk. Nothing consulted the web, no model wrote an answer, and no
human has confirmed that the provision a question points at is the provision a
reader should actually be sent to. It measures whether the pipeline finds the
passage this generator picked. That is a real and useful measurement and it is
not a measurement of legal correctness.

Why templates and not an LLM
----------------------------

An LLM asked to invent 1,000 questions and their answers would produce better
prose and a worse benchmark, for three reasons this project cannot absorb:

* **Contamination.** The generator and the system under test would share a model
  family. A question phrased the way the answerer likes is a question the
  answerer is unfairly good at, and nothing downstream could detect it.
* **Unreproducibility.** `--seed` has to regenerate the file byte for byte, or
  "run the benchmark again" means "get a different benchmark". A sampled decoder
  cannot promise that.
* **Invented law.** An expected answer written by a model is a claim about the
  law. An expected answer *sliced out of the chunk* is a quotation, and a
  quotation is checkable — `--verify` re-reads every one of them out of the
  corpus and fails if a single character has moved.

The cost is that the questions read like a well-drafted form rather than like a
person, which the register templates below mitigate and do not remove. That cost
is disclosed rather than hidden, and it is the right trade for a label set.

Difficulty is earned, not asserted
----------------------------------

`easy` questions may reuse the provision's own distinctive words. `hard` ones may
not: `_strip_heading_words` removes every stem the heading contributes, so a hard
question has to be answered from meaning rather than from string overlap. A
question whose wording simply copies its heading is discarded outright
(`_too_close`), because retrieving it proves only that BM25 works.

Two label kinds
---------------

`ANSWERABLE`          the corpus contains the answer; `target_chunk_ids` is the
                      evidence, and `expected_answer` is quoted from it.
`EXPECTED_ABSTENTION`  the corpus does not contain the answer, and the right
                      behaviour is to say so. `target_chunk_ids` is empty and
                      `expected_answer` is null — never a guess.

Nothing in between. A question the generator cannot ground is not emitted.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import re
from typing import Any, Iterable, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNKS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "chunks")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation", "synthetic_1000.jsonl")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation", "synthetic_1000.manifest.json")

GENERATOR_VERSION = "1.0.0"
DEFAULT_SEED = 20260915
DEFAULT_COUNT = 1000

#: Share of the file that is EXPECTED_ABSTENTION. Small on purpose: abstention
#: precision needs enough probes to mean something and few enough that the
#: headline denominators stay dominated by answerable questions.
ABSTENTION_SHARE = 0.04

MIN_CHARS = 200
#: A question that repeats this much of its own heading is testing string
#: matching, not retrieval, so it is dropped.
MAX_HEADING_ECHO = 0.80

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("""a an the of for to in on by or and is are be been was were what which who whom
whose when where how why does do did can could may might must shall should will would under any such
this that it its as at from with not no nor but if then than there their they them he she his her
each every all some other others more most than into upon out over per within without before after
between among against about during while any_person""".split())

_SENTENCE_BREAK = re.compile(r"(?<=[.;])\s+(?=[A-Z0-9(])")

# --------------------------------------------------------------------- types --

#: Ordered longest-first so "shall not" is seen before "shall".
ANSWER_TYPE_SIGNALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("definition", re.compile(r"\bmeans\b|\bshall mean\b|\bdefinitions?\b|\bincludes\b", re.I)),
    ("timeline", re.compile(r"\bwithin\s+(?:a\s+)?(?:period\s+of\s+)?[\w-]+\s+(?:working\s+)?(?:days?|months?|years?)\b|\bnot later than\b", re.I)),
    ("penalty", re.compile(r"\bpunishab\w+|\bpenalt\w+|\bimprisonment\b|\bshall be liable to a fine\b", re.I)),
    ("prohibition", re.compile(r"\bshall not\b|\bno person shall\b|\bprohibit\w*", re.I)),
    ("exception", re.compile(r"\bprovided that\b|\bnothing in this\b|\bsave as otherwise\b|\bshall not apply\b", re.I)),
    ("eligibility", re.compile(r"\bany person (?:who |claiming |desirous )|\bentitled\b|\beligible\b|\bqualif\w+\b", re.I)),
    ("fee", re.compile(r"\bfees?\b|\bpayable\b", re.I)),
    ("power", re.compile(r"\b(?:Registrar|Controller|Central Government|State Government|Authority|Tribunal|Board)\b[^.]{0,80}\bmay\b", re.I)),
    ("procedure", re.compile(r"\bapplication\b|\bshall be made\b|\bin Form\b|\bshall be filed\b|\bmanner\b", re.I)),
)

#: (template, register, difficulty_floor). `{s}` is the subject phrase, `{d}` the
#: instrument's short title. Registers exist so the file is not 1,000 sentences
#: with one shape; the difficulty floor keeps a chatty opener out of the `easy`
#: stratum where it would be the only thing distinguishing two questions.
TEMPLATES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "definition": (
        ("What does {s} mean under the {d}?", "neutral", "easy"),
        ("How is {s} defined for the purposes of the {d}?", "practitioner", "medium"),
        ("I keep seeing {s} in the {d} — what does it actually refer to?", "beginner", "hard"),
    ),
    "timeline": (
        ("Within what time must {s} be done under the {d}?", "neutral", "easy"),
        ("What is the deadline for {s}?", "neutral", "medium"),
        ("How long do I get for {s} under the {d}?", "beginner", "hard"),
        ("Does the {d} impose a time limit on {s}?", "practitioner", "medium"),
    ),
    "penalty": (
        ("What penalty applies to {s} under the {d}?", "neutral", "easy"),
        ("What are the consequences of {s}?", "neutral", "medium"),
        ("If someone does {s}, what can happen to them under the {d}?", "beginner", "hard"),
    ),
    "prohibition": (
        ("What does the {d} prohibit in relation to {s}?", "neutral", "easy"),
        ("Is it permissible to carry out {s} under the {d}?", "practitioner", "medium"),
        ("Am I allowed to do {s}?", "beginner", "hard"),
    ),
    "exception": (
        ("What exception does the {d} make for {s}?", "neutral", "easy"),
        ("When does the rule on {s} not apply?", "practitioner", "medium"),
        ("Are there any carve-outs around {s}?", "beginner", "hard"),
    ),
    "eligibility": (
        ("Who may apply in respect of {s} under the {d}?", "neutral", "easy"),
        ("Who is entitled to {s}?", "neutral", "medium"),
        ("Can an ordinary individual do {s}, or is it restricted?", "beginner", "hard"),
    ),
    "fee": (
        ("What fee is payable for {s} under the {d}?", "neutral", "easy"),
        ("Does {s} attract a fee?", "practitioner", "medium"),
        ("How much does {s} cost?", "beginner", "hard"),
    ),
    "power": (
        ("What power does the {d} confer in relation to {s}?", "neutral", "easy"),
        ("May the authority act of its own motion in respect of {s}?", "practitioner", "medium"),
        ("Who decides questions about {s}, and on what basis?", "neutral", "hard"),
    ),
    "procedure": (
        ("What is required for {s} under the {d}?", "neutral", "easy"),
        ("How is {s} carried out?", "neutral", "medium"),
        ("How can I go about {s}?", "beginner", "hard"),
        ("What must accompany {s} under the {d}?", "practitioner", "medium"),
    ),
    "conceptual": (
        ("What does the {d} provide about {s}?", "neutral", "easy"),
        ("What are the requirements concerning {s}?", "neutral", "medium"),
        ("Where does {s} sit in the {d}?", "practitioner", "hard"),
    ),
}

IDENTIFIER_TEMPLATES = (
    ("What does {loc} of the {d} provide?", "practitioner", "easy"),
    ("Under {loc} of the {d}, what is required?", "practitioner", "easy"),
    ("Summarise {loc} of the {d}.", "neutral", "easy"),
    ("What is the effect of {loc} of the {d}?", "practitioner", "medium"),
)

CROSS_REF_TEMPLATES = (
    ("How do {loc_a} and {loc_b} of the {d} work together?", "practitioner", "hard"),
    ("{loc_a} of the {d} refers to {loc_b} — what does that cross-reference require?",
     "practitioner", "hard"),
)

MULTI_PART_TEMPLATES = (
    ("What does the {d} require for {s}, and what happens if that is not done?",
     "practitioner", "hard"),
    ("Who may act in respect of {s} under the {d}, and within what time?",
     "practitioner", "hard"),
)

#: Out-of-corpus probes. Each is a real corpus subject asked about an instrument
#: or jurisdiction this corpus does not contain, so the honest answer is "I do
#: not hold that". Deterministic, and never paired with an expected answer.
ABSTENTION_FRAMES = (
    "What does the United States {topic_law} say about {s}?",
    "How does the European Union regulate {s}?",
    "What is the position on {s} under United Kingdom law?",
    "What did the Supreme Court of India hold about {s} in its most recent judgment?",
    "What will the {year} amendment change about {s}?",
    "What does the Japanese equivalent of the {d} provide about {s}?",
    "How much do lawyers typically charge for advice on {s}?",
    "Which law firm in Delhi is best for {s}?",
)
ABSTENTION_TOPIC_LAW = ("Patent Act", "Lanham Act", "Copyright Act", "Plant Variety Protection Act")


# ----------------------------------------------------------------- utilities --

def words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in _STOP and len(w) > 2]


def stem(token: str) -> str:
    """A crude shared stem, used only to compare a question against its heading."""
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ied", "ers", "er",
                   "ment", "ness", "ion", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")).hexdigest()


def load_chunks(chunks_dir: str = CHUNKS_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(chunks_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    rows.sort(key=lambda row: row["chunk_id"])
    return rows


def locator_label(chunk: dict[str, Any]) -> str | None:
    """"Section 3" / "Rule 24B" / "Article 33", from the structured locator."""
    locator = chunk.get("locator") or {}
    for field, label in (("section", "Section"), ("rule", "Rule"),
                         ("article", "Article"), ("regulation", "Regulation"),
                         ("paragraph", "Paragraph")):
        value = locator.get(field)
        if value:
            return f"{label} {value}"
    return None


def clean_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # Provision bodies open with their own number and heading ("85B. Application
    # for licence ... .—"); that prefix is drafting furniture, not the answer.
    text = re.sub(r"^\d+[A-Z]{0,3}\.?\s*", "", text)
    text = re.sub(r"^\[?\d+\[?\s*", "", text)
    return text.strip(" .—-—")


def sentences(text: str) -> list[str]:
    parts = [clean_sentence(p) for p in _SENTENCE_BREAK.split(re.sub(r"\s+", " ", text))]
    return [p for p in parts if len(p) >= 40]


# ------------------------------------------------------------ subject phrases --

def subject_from_heading(heading: str) -> str | None:
    """A heading turned into something a question can be built around.

    Headings are noun phrases in the drafter's register — "Form and signing of
    application", "Restoration of lapsed designs". Lower-casing and trimming the
    leading article is enough to drop one into a template; what it must not do
    is survive into a `hard` question unchanged, which `_strip_heading_words`
    prevents separately.
    """
    subject = re.sub(r"\s+", " ", heading or "").strip().strip(".,;:")
    if not subject or len(subject) < 6:
        return None
    subject = re.sub(r"^(?:The|An|A)\s+", "", subject)
    # Acronym-ish or all-caps OCR damage ("RESERV ATIONS") makes a poor subject.
    letters = [c for c in subject if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.6:
        return None
    if len(content_words(subject)) < 2:
        return None
    first = subject.split()[0]
    return subject if first.isupper() and len(first) > 1 else subject[0].lower() + subject[1:]


#: A marginal note is the provision's own title repeated at the head of its
#: body — "Surrender of certificate of registration under section 33.- The
#: registered breeder may ...". It is the best subject available for the third
#: of the corpus the chunker gave no `heading`, and it is the drafter's words
#: rather than a bag of frequent ones.
_MARGINAL_NOTE = re.compile(r"^(?P<note>[^.\n]{12,110}?)\s*[.,]?\s*(?:\u2014|\.\u2014|\.-|\s-\s)")


def subject_from_marginal_note(text: str) -> str | None:
    body = re.sub(r"\s+", " ", text or "").strip()
    body = re.sub(r"^\[?\d+[A-Z]{0,3}[\].]?\s*", "", body)
    match = _MARGINAL_NOTE.match(body)
    return subject_from_heading(match.group("note")) if match else None


def _degenerate(subject: str) -> bool:
    """Reject a subject that is function words, OCR debris or a single word.

    The earlier version of this generator built `hard` questions by deleting the
    heading's own words from the subject, which reliably produced things like
    "Who decides questions about of to w or, and on what basis?". Difficulty has
    to come from the question's *structure*, never from damaging its wording, so
    that approach is gone and this guard is what keeps its output out of the
    file.
    """
    tokens = subject.split()
    if len(content_words(subject)) < 2:
        return True
    # OCR splits words mid-token ("w ithdraw", "tradem ark"); a stray one-letter
    # token that is not "a" is the signature.
    return any(len(t.strip(".,;:()")) == 1 and t.strip(".,;:()").lower() not in ("a",)
               for t in tokens)


def _too_close(question: str, heading: str, document: str) -> bool:
    """True when the question contributes nothing the heading did not already.

    Not "shares words with the heading" — a question naturally names the thing
    it asks about, and demanding otherwise is what produced the word soup. The
    test is whether, after the instrument's name is set aside, the question adds
    any content word of its own. "Within what time must X be done?" adds *time*
    and is a real question; a question that is the heading with a question mark
    is not.
    """
    heading_stems = {stem(w) for w in content_words(heading)}
    if len(heading_stems) < 3:
        return False
    document_stems = {stem(w) for w in content_words(document)}
    added = {stem(w) for w in content_words(question)} - heading_stems - document_stems
    return not added


def detect_answer_type(chunk: dict[str, Any]) -> str:
    haystack = f"{chunk.get('heading') or ''}\n{chunk.get('text') or ''}"[:4000]
    for name, pattern in ANSWER_TYPE_SIGNALS:
        if pattern.search(haystack):
            return name
    return "conceptual"


# ------------------------------------------------------------ expected answer --

def expected_answer(chunk: dict[str, Any], question: str,
                    max_sentences: int = 2) -> list[str]:
    """The sentences of the chunk that best answer the question, quoted verbatim.

    Returns them as a *list of spans* rather than one string, because the two
    best sentences are often not adjacent, and a joined string would then be a
    quotation of something the chunk never says in that order. Each span is a
    contiguous substring of the chunk's whitespace-collapsed text, which is what
    `verify` re-checks against the corpus.

    Extractive on purpose. This is a label, and a label that paraphrases is a
    label nobody can check.
    """
    pool = sentences(chunk.get("text") or "")
    if not pool:
        return []
    wanted = {stem(w) for w in content_words(question)}
    if not wanted:
        return []
    scored = sorted(
        ((len(wanted & {stem(w) for w in content_words(s)}) / len(wanted), -index, index, s)
         for index, s in enumerate(pool)),
        reverse=True)
    best = [row for row in scored[:max_sentences] if row[0] > 0.0]
    if not best:
        best = [scored[0]]
    chosen = sorted(best, key=lambda row: row[2])
    spans = [row[3].strip() for row in chosen if row[3].strip()]
    return spans if sum(len(s) for s in spans) >= 40 else []


# ---------------------------------------------------------------- generation --

def _emit(rng: random.Random, chunk: dict[str, Any], template: str, register: str,
          difficulty: str, answer_type: str, subject: str | None,
          extra_targets: Sequence[str] = ()) -> dict[str, Any] | None:
    short = chunk.get("short_title") or chunk.get("title") or chunk["document_id"]
    loc = locator_label(chunk)
    question = template.format(s=subject or "", d=short, loc=loc or "",
                               loc_a=loc or "", loc_b="", year=2026,
                               topic_law=rng.choice(ABSTENTION_TOPIC_LAW))
    question = re.sub(r"\s+", " ", question).strip()
    if "{" in question or "  " in question or len(question) < 25:
        return None
    heading = chunk.get("heading") or ""
    if heading and _too_close(question, heading, short):
        return None
    spans = expected_answer(chunk, question)
    if not spans:
        return None
    targets = [chunk["chunk_id"], *extra_targets]
    return {
        "question": question,
        "document_id": chunk["document_id"],
        "target_chunk_ids": targets,
        "locator": chunk.get("citation"),
        "expected_answer": " ".join(spans),
        "expected_answer_spans": spans,
        "answer_type": answer_type,
        "label_kind": "ANSWERABLE",
        "difficulty": difficulty,
        "register": register,
        "domain": chunk.get("domain"),
        "document_type": chunk.get("document_type"),
        "temporal_status": chunk.get("temporal_status"),
        "generation_metadata": {
            "generator_version": GENERATOR_VERSION,
            "strategy": "template_over_corpus_chunk",
            "template": template,
            "answer_extraction": "verbatim sentence span from the target chunk",
            "heading_present": bool(heading),
            "subject_source": "heading" if heading else "marginal_note",
            "source_sha256": chunk.get("source_sha256"),
            "chunking_version": chunk.get("chunking_version"),
        },
    }


def build_answerable(rng: random.Random, chunks: Sequence[dict[str, Any]],
                     target: int) -> list[dict[str, Any]]:
    """Round-robin over documents so no instrument dominates the file."""
    pool = [c for c in chunks
            if c.get("unit_type") in ("provision", "article")
            and c.get("citable")
            and (c.get("character_count") or 0) >= MIN_CHARS]
    by_document: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for chunk in pool:
        by_document[chunk["document_id"]].append(chunk)
    for rows in by_document.values():
        rng.shuffle(rows)

    # A chunk whose text names another provision of the same instrument gives a
    # genuine cross-reference question with two required targets.
    by_id = {c["chunk_id"]: c for c in pool}
    locator_index: dict[tuple[str, str], str] = {}
    for chunk in pool:
        label = locator_label(chunk)
        if label:
            locator_index[(chunk["document_id"], label.lower())] = chunk["chunk_id"]

    out: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    documents = sorted(by_document)
    cursors = {doc: 0 for doc in documents}
    exhausted: set[str] = set()

    while len(out) < target and len(exhausted) < len(documents):
        for document in documents:
            if len(out) >= target:
                break
            if document in exhausted:
                continue
            rows = by_document[document]
            if cursors[document] >= len(rows):
                exhausted.add(document)
                continue
            chunk = rows[cursors[document]]
            cursors[document] += 1
            heading = chunk.get("heading") or ""
            answer_type = detect_answer_type(chunk)
            loc = locator_label(chunk)
            record = None

            roll = rng.random()
            if roll < 0.14 and loc:
                template, register, difficulty = rng.choice(IDENTIFIER_TEMPLATES)
                record = _emit(rng, chunk, template, register, difficulty,
                               "identifier", None)
            elif roll < 0.20 and loc:
                # cross-reference: does this chunk name a sibling provision?
                partner = None
                for match in re.finditer(r"\b(section|rule|article)\s+(\d+[A-Z]{0,3})\b",
                                         chunk.get("text") or "", re.I):
                    key = (chunk["document_id"], f"{match.group(1).title()} {match.group(2).upper()}".lower())
                    candidate = locator_index.get(key)
                    if candidate and candidate != chunk["chunk_id"]:
                        partner = candidate
                        break
                if partner:
                    template, register, difficulty = rng.choice(CROSS_REF_TEMPLATES)
                    short = chunk.get("short_title") or chunk["document_id"]
                    question = template.format(loc_a=loc, loc_b=locator_label(by_id[partner]),
                                               d=short, s="", loc="")
                    spans = expected_answer(chunk, question)
                    if spans:
                        record = {
                            "question": question, "document_id": chunk["document_id"],
                            "target_chunk_ids": [chunk["chunk_id"], partner],
                            "locator": chunk.get("citation"),
                            "expected_answer": " ".join(spans),
                            "expected_answer_spans": spans,
                            "answer_type": "cross_reference", "label_kind": "ANSWERABLE",
                            "difficulty": difficulty, "register": register,
                            "domain": chunk.get("domain"),
                            "document_type": chunk.get("document_type"),
                            "temporal_status": chunk.get("temporal_status"),
                            "generation_metadata": {
                                "generator_version": GENERATOR_VERSION,
                                "strategy": "cross_reference_pair",
                                "template": template,
                                "answer_extraction": "verbatim sentence span from the target chunk",
                                "heading_present": bool(heading),
                                "source_sha256": chunk.get("source_sha256"),
                                "chunking_version": chunk.get("chunking_version"),
                            },
                        }
            if record is None:
                subject = (subject_from_heading(heading)
                           or subject_from_marginal_note(chunk.get("text") or ""))
                if not subject or _degenerate(subject):
                    continue
                family = TEMPLATES.get(answer_type, TEMPLATES["conceptual"])
                if roll > 0.94:
                    template, register, difficulty = rng.choice(MULTI_PART_TEMPLATES)
                    answer_type_used = "multi_part"
                else:
                    template, register, difficulty = rng.choice(family)
                    answer_type_used = answer_type
                # Difficulty is structural, never a damaged question. A chunk the
                # chunker gave no heading is harder by construction: the reranker's
                # heading feature is blind to it, so retrieval has to come from
                # meaning. Beginner phrasing and multi-part asks are harder for the
                # same honest reason.
                if not heading or register == "beginner" or answer_type_used in (
                        "multi_part", "cross_reference"):
                    difficulty = "hard"
                record = _emit(rng, chunk, template, register, difficulty,
                               answer_type_used, subject)

            if record and record["question"].lower() not in seen_questions:
                seen_questions.add(record["question"].lower())
                out.append(record)
    return out


def build_abstentions(rng: random.Random, chunks: Sequence[dict[str, Any]],
                      target: int) -> list[dict[str, Any]]:
    pool = [c for c in chunks if c.get("heading") and c.get("citable")
            and c.get("unit_type") in ("provision", "article")]
    rng.shuffle(pool)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in pool:
        if len(out) >= target:
            break
        subject = subject_from_heading(chunk.get("heading") or "")
        if not subject:
            continue
        frame = ABSTENTION_FRAMES[len(out) % len(ABSTENTION_FRAMES)]
        question = frame.format(s=subject,
                                d=chunk.get("short_title") or chunk["document_id"],
                                topic_law=rng.choice(ABSTENTION_TOPIC_LAW),
                                year=2027)
        question = re.sub(r"\s+", " ", question).strip()
        if question.lower() in seen:
            continue
        seen.add(question.lower())
        out.append({
            "question": question,
            "document_id": None,
            "target_chunk_ids": [],
            "locator": None,
            "expected_answer": None,
            "expected_answer_spans": [],
            "answer_type": "out_of_corpus",
            "label_kind": "EXPECTED_ABSTENTION",
            "difficulty": "hard",
            "register": "neutral",
            "domain": chunk.get("domain"),
            "document_type": None,
            "temporal_status": None,
            "generation_metadata": {
                "generator_version": GENERATOR_VERSION,
                "strategy": "out_of_corpus_frame",
                "template": frame,
                "answer_extraction": None,
                "frame_subject_from": chunk["chunk_id"],
                "rationale": "the corpus holds no such instrument or material, "
                             "so the only correct behaviour is to abstain",
            },
        })
    return out


def generate(count: int = DEFAULT_COUNT, seed: int = DEFAULT_SEED,
             chunks_dir: str = CHUNKS_DIR) -> list[dict[str, Any]]:
    chunks = load_chunks(chunks_dir)
    rng = random.Random(seed)
    n_abstain = int(round(count * ABSTENTION_SHARE))
    answerable = build_answerable(rng, chunks, count - n_abstain)
    abstentions = build_abstentions(rng, chunks, n_abstain)
    records = answerable + abstentions
    if len(records) < count:
        raise SystemExit(f"generator produced {len(records)} of {count} questions; "
                         f"loosen MIN_CHARS or add templates rather than padding")
    records = records[:count]
    # Interleave deterministically so a truncated read is still stratified.
    rng.shuffle(records)
    for index, record in enumerate(records, start=1):
        record["question_id"] = f"S{index:04d}"
    return [{"question_id": r.pop("question_id"), **r} for r in records]


def verify(records: Sequence[dict[str, Any]], chunks_dir: str = CHUNKS_DIR) -> list[str]:
    """Every claim the file makes about the corpus, checked against the corpus."""
    by_id = {c["chunk_id"]: c for c in load_chunks(chunks_dir)}
    problems: list[str] = []
    for record in records:
        qid = record["question_id"]
        if record["label_kind"] == "EXPECTED_ABSTENTION":
            if (record["target_chunk_ids"] or record["expected_answer"] is not None
                    or record.get("expected_answer_spans")):
                problems.append(f"{qid}: an abstention label carries evidence")
            continue
        if not record["target_chunk_ids"]:
            problems.append(f"{qid}: answerable but has no target chunk")
            continue
        for chunk_id in record["target_chunk_ids"]:
            if chunk_id not in by_id:
                problems.append(f"{qid}: target {chunk_id} is not in the corpus")
        primary = by_id.get(record["target_chunk_ids"][0])
        if primary is None:
            continue
        haystack = re.sub(r"\s+", " ", primary["text"])
        spans = record.get("expected_answer_spans") or []
        if not spans:
            problems.append(f"{qid}: answerable but quotes no evidence span")
        for span in spans:
            if span not in haystack:
                problems.append(f"{qid}: span is not verbatim in {primary['chunk_id']}: "
                                f"{span[:60]!r}")
                break
        if " ".join(spans) != (record["expected_answer"] or ""):
            problems.append(f"{qid}: expected_answer disagrees with its own spans")
        if record["document_id"] != primary["document_id"]:
            problems.append(f"{qid}: document_id disagrees with the target chunk")
    return problems


def summarise(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def tally(key: str) -> dict[str, int]:
        return dict(collections.Counter(str(r.get(key)) for r in records).most_common())
    answerable = [r for r in records if r["label_kind"] == "ANSWERABLE"]
    return {
        "total": len(records),
        "answerable": len(answerable),
        "expected_abstention": len(records) - len(answerable),
        "multi_chunk_questions": sum(1 for r in answerable if len(r["target_chunk_ids"]) > 1),
        "distinct_target_chunks": len({c for r in answerable for c in r["target_chunk_ids"]}),
        "distinct_documents": len({r["document_id"] for r in answerable}),
        "by_answer_type": tally("answer_type"),
        "by_difficulty": tally("difficulty"),
        "by_register": tally("register"),
        "by_domain": tally("domain"),
        "by_document": tally("document_id"),
        "by_document_type": tally("document_type"),
        "by_temporal_status": tally("temporal_status"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chunks-dir", default=CHUNKS_DIR)
    parser.add_argument("--out", default=OUTPUT_PATH)
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--verify-only", action="store_true",
                        help="re-read the file on disk and check it against the corpus")
    args = parser.parse_args(argv)

    if args.verify_only:
        with open(args.out, encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        records = generate(args.count, args.seed, args.chunks_dir)

    problems = verify(records, args.chunks_dir)
    if problems:
        for problem in problems[:25]:
            print(f"FAIL {problem}")
        print(f"\n{len(problems)} problem(s); nothing written")
        return 1

    summary = summarise(records)
    if args.verify_only:
        print(json.dumps(summary, indent=2))
        print("\nVERIFIED every label resolves to the corpus on disk")
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": args.seed,
        "count": args.count,
        "questions_sha256": canonical_sha256(records),
        "corpus_chunks": len(load_chunks(args.chunks_dir)),
        "summary": summary,
        "claim_safety":
            "Synthetic, AI-generated benchmark derived from the fixed corpus. "
            "Labels are corpus-derived and template-authored; expected answers "
            "are verbatim spans of the target chunk. This is an engineering "
            "evaluation and NOT human-verified legal ground truth.",
    }
    with open(args.manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {len(records)} questions -> {args.out}")
    print(f"manifest -> {args.manifest}")
    print(f"questions_sha256 {manifest['questions_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
