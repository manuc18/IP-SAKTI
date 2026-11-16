"""IP-SAKTI corpus audit. READ-ONLY: opens files in 'rb' only, writes nothing to the corpus."""
import hashlib, json, os, re, sys, unicodedata
from collections import Counter

from pypdf import PdfReader
from pypdf.errors import PdfReadError

CORPUS = os.path.expanduser("~/Downloads/IP-SAKTI Corpus")
OUT = sys.argv[1]

STRUCT_PATTERNS = {
    "act_title":   r"(?im)^\s*(THE\s+)?[A-Z][A-Za-z'’\(\)\s,\-]{6,90}\s+(ACT|RULES|REGULATIONS|PROTOCOL|CONVENTION|AGREEMENT|POLICY)\s*,?\s*\d{4}",
    "chapter":     r"(?im)\bCHAPTER\s+([IVXLCDM]+|\d+)\b",
    "section":     r"(?im)(\bSECTION\s+\d+|^\s*\d{1,3}\.\s+[A-Z])",
    "subsection":  r"\(\s*\d{1,2}\s*\)",
    "clause":      r"(?im)(\bclause\s*\(\s*[a-z]{1,3}\s*\)|^\s*\(\s*[a-z]{1,3}\s*\)\s)",
    "rule":        r"(?im)\bRules?\s+\d{1,3}[A-Z]?\b",
    "sub_rule":    r"(?im)\bsub-?rule\s*\(\s*\d+\s*\)",
    "schedule":    r"(?im)\b(THE\s+)?(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH)?\s*SCHEDULE\b",
    "annexure":    r"(?im)\bANNEX(URE|)\s*[-–]?\s*([IVXLCDM]+|\d+|[A-Z])?\b",
    "article":     r"(?im)\bARTICLE\s+(\d+|[IVXLCDM]+)\b",
    "regulation":  r"(?im)\bRegulations?\s+\d{1,3}",
}
URL_RE = re.compile(r"(?i)\b((?:https?://|www\.)[^\s<>\"'\)\]]{4,120})")
DOMAIN_RE = re.compile(r"(?i)\b([a-z0-9][a-z0-9\-]{1,60}\.(?:gov\.in|nic\.in|org|int|com|net|in))\b")
ACTNO_RE = re.compile(r"(?i)\bACT\s+No\.?\s*([0-9]{1,3})\s+OF\s+((?:19|20)\d{2})")
GSR_RE = re.compile(r"(?i)\b(G\.?\s?S\.?\s?R\.?|S\.?\s?O\.?)\s*([0-9]{1,5}\s*\(?[EA]?\)?)")
DATE_RE = re.compile(r"(?i)\b(\d{1,2}(?:st|nd|rd|th)?[\s\-/]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-/,]+(?:19|20)\d{2})")
CID_RE = re.compile(r"\(cid:\d+\)")
PAGENUM_RE = re.compile(r"(?m)^\s*(?:page\s*)?\d{1,4}\s*(?:of\s*\d{1,4})?\s*$", re.I)

EN_STOP = {"the","and","of","to","in","for","or","be","shall","any","such","as","by","with","which","that","under","this","on","is","not"}
HI_MARK = "ऀॿ"

def script_ratios(t):
    dev = lat = tot = 0
    for ch in t:
        if ch.isspace() or not ch.isprintable():
            continue
        tot += 1
        o = ord(ch)
        if 0x0900 <= o <= 0x097F: dev += 1
        elif ('a' <= ch <= 'z') or ('A' <= ch <= 'Z'): lat += 1
    return dev, lat, tot

def detect_language(t):
    dev, lat, tot = script_ratios(t)
    if tot < 50: return None, {}
    ev = {"devanagari_ratio": round(dev/tot, 4), "latin_ratio": round(lat/tot, 4)}
    words = re.findall(r"[a-z]+", t.lower())
    stop_hits = sum(1 for w in words if w in EN_STOP)
    ev["en_stopword_ratio"] = round(stop_hits/len(words), 4) if words else 0.0
    if dev/tot > 0.30: lang = "hi" if lat/tot < 0.15 else "en+hi"
    elif ev["en_stopword_ratio"] > 0.05: lang = "en"
    elif lat/tot > 0.6: lang = "en?"
    else: lang = None
    return lang, ev

def garble_score(t):
    if not t: return 0.0, {}
    cid = len(CID_RE.findall(t))
    repl = t.count("�")
    ctrl = sum(1 for c in t if unicodedata.category(c) in ("Cc","Co","Cn") and c not in "\n\r\t")
    printable = sum(1 for c in t if c.isprintable() or c.isspace())
    ratio = 1 - (printable/len(t))
    # vowel-less gibberish check on ascii words
    words = re.findall(r"[A-Za-z]{4,}", t)
    novowel = sum(1 for w in words[:5000] if not re.search(r"[aeiouAEIOU]", w))
    nv = round(novowel/len(words[:5000]), 4) if words else 0.0
    score = min(1.0, (cid*3 + repl*3 + ctrl) / max(len(t), 1) * 100 + (nv if nv > 0.25 else 0))
    return round(score, 4), {"cid_artifacts": cid, "replacement_chars": repl,
                             "control_chars": ctrl, "nonprintable_ratio": round(ratio, 5),
                             "vowelless_word_ratio": nv}

def table_score(page_text):
    lines = [l for l in page_text.split("\n") if l.strip()]
    if not lines: return 0.0
    hits = sum(1 for l in lines if len(re.findall(r"\S {2,}\S", l)) >= 2 or l.count("|") >= 2 or l.count("\t") >= 2)
    return round(hits/len(lines), 3)

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def audit_pdf(path):
    r = {}
    with open(path, "rb") as fh:
        head = fh.read(1024)
        fh.seek(0)
        raw = fh.read()
    r["magic"] = head[:8].decode("latin-1", "replace")
    r["pdf_version_header"] = head[:8].decode("latin-1","replace").strip() if head.startswith(b"%PDF") else None
    r["has_eof_marker"] = raw.rstrip()[-20:].find(b"%%EOF") >= 0
    r["sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        reader = PdfReader(path)
    except Exception as e:
        r["open_error"] = f"{type(e).__name__}: {e}"
        return r
    r["encrypted"] = bool(reader.is_encrypted)
    if reader.is_encrypted:
        try: reader.decrypt("")
        except Exception as e: r["decrypt_error"] = str(e)
    try: r["page_count"] = len(reader.pages)
    except Exception as e:
        r["page_count"] = None; r["open_error"] = str(e); return r
    md = {}
    try:
        meta = reader.metadata or {}
        for k, v in meta.items():
            try: md[str(k)] = str(v)
            except Exception: pass
    except Exception as e:
        md["_error"] = str(e)
    r["pdf_metadata"] = md
    try: r["outline_count"] = len(reader.outline or [])
    except Exception: r["outline_count"] = None

    pages, errs = [], []
    for i in range(r["page_count"]):
        try:
            t = reader.pages[i].extract_text() or ""
        except Exception as e:
            t = ""; errs.append({"page": i+1, "error": f"{type(e).__name__}: {e}"})
        try: imgs = len(reader.pages[i].images)
        except Exception: imgs = None
        pages.append({"n": i+1, "chars": len(t.strip()), "text": t, "images": imgs,
                      "table_score": table_score(t)})
    r["page_errors"] = errs[:20]
    r["page_error_count"] = len(errs)

    full = "\n".join(p["text"] for p in pages)
    r["extracted_chars"] = len(full.strip())
    r["chars_per_page"] = round(r["extracted_chars"]/max(r["page_count"],1), 1)
    empty = [p["n"] for p in pages if p["chars"] == 0]
    thin  = [p["n"] for p in pages if 0 < p["chars"] < 100]
    r["empty_text_pages"] = empty
    r["empty_text_page_count"] = len(empty)
    r["thin_text_pages_count"] = len(thin)
    r["image_only_pages"] = [p["n"] for p in pages if p["chars"] < 100 and (p["images"] or 0) > 0]
    r["total_embedded_images"] = sum(p["images"] or 0 for p in pages)
    r["text_extractable"] = r["extracted_chars"] > 500
    r["ocr_pages"] = sorted(set(empty + thin))
    r["ocr_page_fraction"] = round(len(r["ocr_pages"])/max(r["page_count"],1), 4)
    r["text_native"] = r["text_extractable"] and r["ocr_page_fraction"] < 0.10
    r["ocr_required"] = (not r["text_extractable"]) or r["ocr_page_fraction"] >= 0.10

    r["garble_score"], r["garble_evidence"] = garble_score(full)
    r["language"], r["language_evidence"] = detect_language(full)

    struct = {}
    for k, pat in STRUCT_PATTERNS.items():
        struct[k] = len(re.findall(pat, full))
    r["structural_elements"] = struct
    r["structural_elements_present"] = sorted([k for k, v in struct.items() if v >= 3])

    # headers / footers
    firsts = Counter(); lasts = Counter()
    for p in pages:
        ls = [l.strip() for l in p["text"].split("\n") if l.strip()]
        if ls:
            firsts[re.sub(r"\d+", "#", ls[0])[:80]] += 1
            lasts[re.sub(r"\d+", "#", ls[-1])[:80]] += 1
    npg = max(r["page_count"], 1)
    r["repeated_header_candidates"] = [{"text": t, "pages": c} for t, c in firsts.most_common(4) if c > 2 and c/npg > 0.15]
    r["repeated_footer_candidates"] = [{"text": t, "pages": c} for t, c in lasts.most_common(4) if c > 2 and c/npg > 0.15]
    r["page_numbering_detected"] = sum(1 for p in pages if PAGENUM_RE.search(p["text"])) > npg * 0.3

    r["pages_with_tables"] = [p["n"] for p in pages if p["table_score"] > 0.3]
    r["pages_with_tables_count"] = len(r["pages_with_tables"])

    # duplicate pages
    seen = {}
    dups = []
    for p in pages:
        key = norm(p["text"])
        if len(key) < 200: continue
        h = hashlib.md5(key.encode()).hexdigest()
        if h in seen: dups.append([seen[h], p["n"]])
        else: seen[h] = p["n"]
    r["duplicate_page_pairs"] = dups[:30]
    r["duplicate_page_count"] = len(dups)

    # provenance evidence from text
    front = "\n".join(p["text"] for p in pages[:4])
    back = "\n".join(p["text"] for p in pages[-2:])
    r["urls_in_text"] = sorted(set(URL_RE.findall(full)))[:15]
    r["domains_in_text"] = [d for d, _ in Counter(DOMAIN_RE.findall(full)).most_common(8)]
    r["act_numbers"] = sorted(set(" ".join(m) for m in ACTNO_RE.findall(full)))[:8]
    r["gsr_so_numbers"] = sorted(set(f"{a} {b}".strip() for a, b in GSR_RE.findall(full)))[:10]
    r["dates_in_front_matter"] = sorted(set(DATE_RE.findall(front)))[:10]
    r["dates_in_back_matter"] = sorted(set(DATE_RE.findall(back)))[:10]
    r["years_mentioned_top"] = [y for y, _ in Counter(re.findall(r"\b(19[5-9]\d|20[0-4]\d)\b", full)).most_common(8)]
    r["first_page_text"] = pages[0]["text"][:2500] if pages else ""
    r["front_matter_text"] = front[:6000]
    r["last_page_text"] = pages[-1]["text"][:1200] if pages else ""
    return r


def main():
    out = {"corpus_root": CORPUS, "files": []}
    for dirpath, _, names in os.walk(CORPUS):
        for n in sorted(names):
            p = os.path.join(dirpath, n)
            if os.path.islink(p): continue
            st = os.stat(p)
            rec = {"filename": n, "absolute_path": p,
                   "relative_path": os.path.relpath(p, CORPUS),
                   "extension": os.path.splitext(n)[1].lower(),
                   "file_size_bytes": st.st_size,
                   "mtime": st.st_mtime}
            if rec["extension"] == ".pdf":
                rec["file_class"] = "pdf"
                try:
                    rec.update(audit_pdf(p))
                except Exception as e:
                    rec["fatal_error"] = f"{type(e).__name__}: {e}"
            else:
                rec["file_class"] = "other"
            out["files"].append(rec)
            print(f"done: {n}", file=sys.stderr, flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"WROTE {OUT}", file=sys.stderr)

main()
