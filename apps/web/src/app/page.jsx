"use client";

import React, { useState, useEffect, useRef, useMemo } from "react";
import {
  PlusIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  ExclamationTriangleIcon,
  FileTextIcon,
  StopIcon,
  SpeakerLoudIcon,
  CheckIcon,
  UpdateIcon,
  DotsHorizontalIcon,
  ReloadIcon,
  Pencil1Icon,
  TrashIcon,
  GlobeIcon,
  BookmarkIcon,
  MagnifyingGlassIcon,
  Cross2Icon,
} from "@radix-ui/react-icons";
import {
  Loader2,
  Shield as ShieldIcon,
  Mic as MicrophoneIcon,
  Scale as BalanceIcon,
  Paperclip as PaperclipIcon,
  Lightbulb as LightbulbIcon,
  Search as SearchIcon,
  Zap as ZapIcon,
  Menu,
  Plus,
  Trash2,
  PanelLeft,
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import Stepper, { Step } from "../components/Stepper/Stepper";
import {
  ChatInput,
  ChatInputTextArea,
  ChatInputSubmit,
} from "../components/ui/chat-input";
import { Sidebar } from "../components/Sidebar";
import { BackgroundGradientAnimation } from "../components/ui/background-gradient-animation";
import { getUIText } from "../lib/i18n";
import { LANGUAGES, languageByCode } from "../lib/languages.mjs";
import { collectProtectedTerms } from "../lib/translation-client.mjs";
import { DEMO_QUESTION_TEXTS } from "../lib/demo-questions.mjs";
import {
  queryTranslationFailed,
  resolveQueryLanguage,
} from "../lib/query-language.mjs";
import { describeResult } from "../lib/result-status.mjs";
import {
  selectPreferredVoice,
  voicesForLocale,
} from "../lib/voice-selection.mjs";

// One curated list feeds the tutorial, the landing chips and the rotating
// placeholder. See lib/demo-questions.mjs for why each question is demo-safe.
const DEMO_QUESTIONS = DEMO_QUESTION_TEXTS;

// API endpoints
const PRIMARY_API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001";
const FALLBACK_API_URL = "http://localhost:8000";
const TRANSLATION_FALLBACK =
  "Translation is unavailable right now. You can continue in English.";
const LEGAL_DISCLAIMER =
  "IP-SAKTI is an information and research tool that answers from a fixed corpus of published legal texts. It is not legal advice and is not a substitute for a qualified practitioner.";

async function translateText({
  text,
  sourceLanguage,
  targetLanguage,
  cache = null,
}) {
  if (!text || sourceLanguage === targetLanguage) return text;
  const [translated] = await translateBatch(
    [text],
    targetLanguage,
    sourceLanguage,
    cache,
  );
  return translated || text;
}

/**
 * Translate an array of texts in a SINGLE HTTP call.
 * Returns the translated array in the same order. Nulls/empty strings pass through unchanged.
 * Uses a per-session in-memory cache to avoid re-translating the same text.
 */
async function translateBatch(texts, targetLang, sourceLang = "en", cache = null) {
  if (!Array.isArray(texts) || texts.length === 0) return texts;
  if (targetLang === sourceLang) return texts;

  // Check which need to be fetched (not in cache)
  const results = [...texts];
  const fetchIndexes = [];
  const fetchTexts = [];

  texts.forEach((text, i) => {
    if (!text || typeof text !== "string" || !text.trim()) return; // blank → keep
    if (cache) {
      const key = `${sourceLang}->${targetLang}::${text.trim()}`;
      if (cache.has(key)) {
        results[i] = cache.get(key);
        return;
      }
    }
    fetchIndexes.push(i);
    fetchTexts.push(text);
  });

  if (fetchTexts.length === 0) return results;

  try {
    const response = await fetch("/api/translate-batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts: fetchTexts, sourceLanguage: sourceLang, targetLanguage: targetLang }),
    });
    const data = await response.json().catch(() => ({}));
    if (response.ok && Array.isArray(data.translatedTexts)) {
      data.translatedTexts.forEach((translated, idx) => {
        const originalIdx = fetchIndexes[idx];
        if (translated) {
          results[originalIdx] = translated;
          if (cache && translated.trim() !== texts[originalIdx].trim()) {
            const key = `${sourceLang}->${targetLang}::${texts[originalIdx].trim()}`;
            cache.set(key, translated);
          }
        }
      });
    }
  } catch {
    // Graceful fallback: keep originals
  }

  return results;
}


const ANALYSIS_STEPS = [
  "Query processing",
  "Evidence retrieval",
  "Reranking sources",
  "Selecting evidence",
  "Generating grounded answer",
  "Validating citations",
];

// Isolated source evidence item with dynamic document title translation and excerpt translation
function SourceItem({ src, index, msgId, uiText, selectedLanguageCode, translationCache }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [displayText, setDisplayText] = useState(src.text || "");
  const [isTranslatingText, setIsTranslatingText] = useState(false);
  const [docNameTranslated, setDocNameTranslated] = useState("");
  const marker = src.marker || index + 1;
  const rawLocator = src.locator_text || src.citation || "";
  const formattedLocator = uiText?.formatLocator
    ? uiText.formatLocator(rawLocator)
    : rawLocator;
  const pages = Array.isArray(src.physical_pages) ? src.physical_pages : [];
  const pageLabel = pages.length
    ? pages.length > 1
      ? `pp. ${pages[0]}–${pages[pages.length - 1]}`
      : `p. ${pages[0]}`
    : "";
  const isStale = src.temporal_status && src.temporal_status !== "CURRENT";
  // Detected by ai/text_integrity.py and never repaired: damaged text must not
  // be presented as quoted legal wording.
  const isDamaged = src.text_integrity?.status === "EXTRACTION_DAMAGED";
  const excerptText = src.excerpt?.text || "";

  const rawDocName = src.originalDocumentName || src.document_name || src.document_id || "";

  // 1. Resolve document title synchronously if formatDocumentName has it or if src has it
  const resolvedDocName = useMemo(() => {
    if (!selectedLanguageCode || selectedLanguageCode === "en") return rawDocName;
    if (docNameTranslated) return docNameTranslated;
    if (uiText?.formatDocumentName) {
      const formatted = uiText.formatDocumentName(rawDocName);
      if (formatted && formatted !== rawDocName) return formatted;
    }
    if (src.document_name && src.document_name !== rawDocName) {
      return src.document_name;
    }
    return rawDocName;
  }, [selectedLanguageCode, rawDocName, docNameTranslated, uiText, src.document_name]);

  // Translate document title dynamically if non-English
  useEffect(() => {
    let cancelled = false;
    if (!selectedLanguageCode || selectedLanguageCode === "en") {
      setDocNameTranslated("");
      return;
    }
    if (!rawDocName.trim()) return;

    if (uiText?.formatDocumentName) {
      const formatted = uiText.formatDocumentName(rawDocName);
      if (formatted && formatted !== rawDocName) {
        setDocNameTranslated(formatted);
        return;
      }
    }

    const cacheKey = `en->${selectedLanguageCode}::${rawDocName.trim()}`;
    const cache = translationCache?.current;
    if (cache && cache.has(cacheKey)) {
      setDocNameTranslated(cache.get(cacheKey));
      return;
    }

    translateBatch([rawDocName], selectedLanguageCode, "en", cache).then(([translated]) => {
      if (!cancelled && translated && translated !== rawDocName) {
        setDocNameTranslated(translated);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [selectedLanguageCode, rawDocName, uiText, translationCache]);

  // On expansion or when language changes, translate excerpt if non-English
  useEffect(() => {
    let cancelled = false;

    if (!isExpanded) return;

    if (!selectedLanguageCode || selectedLanguageCode === "en") {
      setDisplayText(src.originalText || src.text || "");
      setIsTranslatingText(false);
      return;
    }

    const rawText = src.originalText || src.text || "";
    if (!rawText.trim()) return;

    const cacheKey = `en->${selectedLanguageCode}::${rawText.trim()}`;
    const cache = translationCache?.current;
    if (cache && cache.has(cacheKey)) {
      setDisplayText(cache.get(cacheKey));
      setIsTranslatingText(false);
      return;
    }

    setIsTranslatingText(true);
    translateBatch([rawText], selectedLanguageCode, "en", cache)
      .then(([translated]) => {
        if (!cancelled && translated) {
          setDisplayText(translated);
        }
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) {
          setIsTranslatingText(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [isExpanded, selectedLanguageCode, src.text, src.originalText, translationCache]);

  return (
    <div
      id={`src-${msgId}-${marker}`}
      className={`border rounded-2xl p-3 sm:p-4 backdrop-blur-sm transition scroll-mt-4 ${src.cited
        ? "border-emerald-200/80 bg-white/90"
        : "border-[#e8edf3] bg-white/70"
        }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-start gap-2 sm:gap-2.5 min-w-0">
          <span
            className={`mt-0.5 shrink-0 inline-flex items-center justify-center min-w-[28px] h-[22px] px-1.5 rounded-md text-[11px] sm:text-xs font-bold ${src.cited
              ? "bg-emerald-600 text-white"
              : "bg-slate-100 text-slate-500 border border-slate-200"
              }`}
          >
            [{marker}]
          </span>
          <div className="space-y-1 min-w-0">
            <h4 className="text-xs sm:text-[13px] font-semibold text-[#0f172a] leading-tight break-words">
              {resolvedDocName}
            </h4>
            <p className="text-[11px] sm:text-xs text-[#0f172a] break-words">
              <span className="font-semibold text-emerald-800">
                {formattedLocator || (uiText?.locatorUnavailable || "Locator unavailable")}
              </span>
              {src.heading && <span className="text-[#475569]"> — {src.heading}</span>}
              {pageLabel && <span className="text-[#94a3b8]"> · {pageLabel}</span>}
            </p>
            <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
              <span
                className={`text-[9px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wide border ${src.cited
                  ? "bg-[#ecfdf5] text-[#047857] border-[#a7f3d0]"
                  : "bg-[#f8fafc] text-[#64748b] border-[#e2e8f0]"
                  }`}
              >
                {src.cited
                  ? (uiText?.citedInAnswer || "Cited in answer")
                  : (uiText?.retrievedNotCited || "Retrieved · not cited")}
              </span>
              {src.temporal_status && (
                <span
                  className={`text-[9px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wide border ${isStale
                    ? "bg-amber-50 text-amber-700 border-amber-200"
                    : "bg-[#ecfdf5] text-[#047857] border-[#a7f3d0]"
                    }`}
                  title={isStale ? "This instrument may have been amended since this copy." : undefined}
                >
                  {isStale ? (uiText?.stale || "Stale") : (uiText?.current || "Current")}
                </span>
              )}
              {src.source_evidence_level && (
                <span
                  className="text-[9px] font-bold px-2 py-0.5 rounded-full bg-[#f8fafc] text-[#475569] border border-[#e2e8f0] uppercase tracking-wide"
                  title="Provenance level of the source copy (A = strongest)"
                >
                  {uiText?.provenance || "Provenance"} {src.source_evidence_level}
                </span>
              )}
              {isDamaged && (
                <span className="text-[9px] font-bold px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200 uppercase tracking-wide">
                  {uiText?.textDamaged || "Text damaged"}
                </span>
              )}
            </div>
          </div>
        </div>

        <button
          onClick={() => setIsExpanded((prev) => !prev)}
          className="flex items-center gap-1 text-[11px] font-medium text-[#64748b] hover:text-[#0f172a] shrink-0 pt-0.5 transition"
        >
          <span>{isExpanded ? (uiText?.hide || "Hide") : (uiText?.viewFullPassage || "Full passage")}</span>
          {isExpanded ? (
            <ChevronUpIcon className="w-3.5 h-3.5" />
          ) : (
            <ChevronDownIcon className="w-3.5 h-3.5" />
          )}
        </button>
      </div>

      {isDamaged ? (
        <div className="mt-2.5 rounded-xl border border-amber-200 bg-amber-50/70 px-3 py-2 text-[11px] leading-relaxed text-amber-800">
          The source text of this passage did not extract cleanly from the PDF, so it is
          not shown as quoted legal wording. Check the original document
          {pageLabel ? ` (${pageLabel})` : ""}.
          {src.official_source_url && (
            <>
              {" "}
              <a
                href={src.official_source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="font-semibold underline"
              >
                Open official source
              </a>
            </>
          )}
        </div>
      ) : (
        excerptText && (
          <blockquote className="mt-2.5 border-l-2 border-emerald-300 pl-3 text-[11px] sm:text-xs leading-relaxed text-slate-700 line-clamp-3 break-words">
            {excerptText}
          </blockquote>
        )
      )}

      {isExpanded && (
        <div className="mt-3 pt-3 border-t border-[#f1f5f9] text-xs text-gray-700 bg-gray-50/80 p-3 rounded-xl leading-relaxed max-h-60 overflow-y-auto">
          {isDamaged && (
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-amber-700">
              Raw extracted text — may be garbled, not authoritative
            </p>
          )}
          {isTranslatingText ? (
            <div className="flex items-center gap-2 text-xs text-slate-500 py-2">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-emerald-600 shrink-0" />
              <span>{uiText?.analyzingEvidence || "Translating source..."}</span>
            </div>
          ) : (
            <p className="whitespace-pre-wrap break-words">{displayText}</p>
          )}
        </div>
      )}
    </div>
  );
}

// Sources for one answer. Cited passages come first, numbered exactly as the
// answer's [n] markers; retrieved-but-uncited passages are folded away so they
// are never mistaken for the answer's authority.
function SourcesList({ msg, uiText, selectedLanguageCode, translationCache }) {
  const [showOthers, setShowOthers] = useState(false);
  const evidence = msg.evidence || [];
  const cited = evidence.filter((src) => src.cited);
  const others = evidence.filter((src) => !src.cited);

  const renderItem = (src, sIdx) => (
    <SourceItem
      key={src.chunk_id || `${msg.id}-src-${sIdx}`}
      src={src}
      index={evidence.indexOf(src)}
      msgId={msg.id}
      uiText={uiText}
      selectedLanguageCode={selectedLanguageCode}
      translationCache={translationCache}
    />
  );

  return (
    <div className="pt-2 border-t border-slate-100 space-y-2">
      <div className="flex flex-wrap items-center gap-1.5 text-slate-400">
        <FileTextIcon className="w-3.5 h-3.5" />
        <span className="text-[9px] sm:text-[10px] font-semibold tracking-widest uppercase">
          {uiText.sources || "Sources"}
        </span>
        <span className="text-[10px] text-slate-400">
          · {cited.length > 0
            ? `${cited.length} cited in answer`
            : `${others.length} retrieved, none cited`}
        </span>
      </div>

      {cited.length > 0 && <div className="space-y-2">{cited.map(renderItem)}</div>}

      {others.length > 0 && (
        <div className="space-y-2">
          <button
            type="button"
            onClick={() => setShowOthers((prev) => !prev)}
            className="flex items-center gap-1 text-[11px] font-medium text-slate-500 hover:text-slate-800 transition"
          >
            {showOthers ? <ChevronUpIcon className="w-3.5 h-3.5" /> : <ChevronDownIcon className="w-3.5 h-3.5" />}
            <span>
              {showOthers ? "Hide" : "Show"} {others.length} other retrieved{" "}
              {others.length === 1 ? "passage" : "passages"} (not cited)
            </span>
          </button>
          {showOthers && <div className="space-y-2">{others.map(renderItem)}</div>}
        </div>
      )}
    </div>
  );
}

// Skeleton loader for assistant messages during language translation
function AssistantMessageSkeleton() {
  return (
    <div className="flex items-start gap-2.5 sm:gap-3 pt-2 scroll-mt-4 animate-pulse">
      <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-slate-200/80 shrink-0 mt-0.5" />
      <div className="flex-1 min-w-0 max-w-[1080px] bg-white/95 border border-slate-200/80 rounded-2xl sm:rounded-3xl p-3.5 sm:p-6 shadow-sm space-y-4">
        {/* Top Badge & Confidence Skeleton */}
        <div className="flex items-center justify-between gap-2">
          <div className="h-4 w-32 bg-slate-200/80 rounded-full" />
          <div className="h-4 w-16 bg-slate-200/80 rounded-full" />
        </div>

        {/* Answer Text Lines Skeleton */}
        <div className="space-y-2.5 py-1">
          <div className="h-3.5 bg-slate-200/80 rounded-full w-full" />
          <div className="h-3.5 bg-slate-200/80 rounded-full w-[94%]" />
          <div className="h-3.5 bg-slate-200/80 rounded-full w-[88%]" />
          <div className="h-3.5 bg-slate-200/80 rounded-full w-[65%]" />
        </div>

        {/* Meta toolbar skeleton */}
        <div className="flex items-center justify-between pt-3 border-t border-slate-100">
          <div className="h-6 w-36 bg-slate-200/80 rounded-full" />
          <div className="h-6 w-24 bg-slate-200/80 rounded-full" />
        </div>

        {/* Sources Skeleton */}
        <div className="pt-2 border-t border-slate-100 space-y-2">
          <div className="h-3 w-20 bg-slate-200/80 rounded-full" />
          <div className="space-y-2">
            <div className="h-16 bg-slate-100/80 border border-slate-200/60 rounded-2xl" />
            <div className="h-16 bg-slate-100/80 border border-slate-200/60 rounded-2xl" />
          </div>
        </div>

        {/* Disclaimer Skeleton */}
        <div className="h-3 bg-slate-200/60 rounded-full w-3/4 pt-1" />
      </div>
    </div>
  );
}

// Skeleton loader for user question during language translation
function UserMessageSkeleton({ isLatestUser, latestTurnRef }) {
  return (
    <div
      ref={isLatestUser ? latestTurnRef : null}
      className="flex items-start justify-end gap-2 sm:gap-3 group pt-1 scroll-mt-4 animate-pulse"
    >
      <div className="flex flex-col items-end max-w-[90%] sm:max-w-lg w-full">
        <div className="bg-white border border-slate-200/80 rounded-2xl sm:rounded-3xl px-4 sm:px-5 py-3.5 shadow-sm w-48 sm:w-72 space-y-2">
          <div className="h-3.5 bg-slate-200/80 rounded-full w-full" />
          <div className="h-3.5 bg-slate-200/70 rounded-full w-3/4 ml-auto" />
        </div>
        <div className="flex items-center gap-3 mt-1.5 text-[11px] text-slate-300">
          <div className="h-3 w-10 bg-slate-200/60 rounded-full" />
          <div className="h-3 w-12 bg-slate-200/60 rounded-full" />
        </div>
      </div>
      <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-slate-200/80 shrink-0" />
    </div>
  );
}

// Dedicated reactive component for user question with automatic language translation
function UserMessageItem({
  msg,
  isLatestUser,
  latestTurnRef,
  uiText,
  selectedLanguageCode,
  translationCache,
  setInputText,
  handleSendMessage,
  isGlobalTranslating = false,
}) {
  const raw = msg.originalContent || msg.content || "";
  const [displayContent, setDisplayContent] = useState(() => {
    if (!selectedLanguageCode || selectedLanguageCode === "en") return raw;
    const cacheKey = `en->${selectedLanguageCode}::${raw.trim()}`;
    const cache = translationCache?.current;
    if (cache && cache.has(cacheKey)) return cache.get(cacheKey);
    return "";
  });
  const [isTranslatingMsg, setIsTranslatingMsg] = useState(() => {
    if (!selectedLanguageCode || selectedLanguageCode === "en") return false;
    const cacheKey = `en->${selectedLanguageCode}::${raw.trim()}`;
    const cache = translationCache?.current;
    if (cache && cache.has(cacheKey)) return false;
    return true;
  });

  useEffect(() => {
    let cancelled = false;
    if (!selectedLanguageCode || selectedLanguageCode === "en") {
      setDisplayContent(raw);
      setIsTranslatingMsg(false);
      return;
    }
    const cacheKey = `en->${selectedLanguageCode}::${raw.trim()}`;
    const cache = translationCache?.current;
    if (cache && cache.has(cacheKey)) {
      setDisplayContent(cache.get(cacheKey));
      setIsTranslatingMsg(false);
      return;
    }

    setIsTranslatingMsg(true);
    translateBatch([raw], selectedLanguageCode, "en", cache)
      .then(([translated]) => {
        if (!cancelled) {
          if (translated && translated !== raw) {
            setDisplayContent(translated);
          } else {
            setDisplayContent(raw);
          }
          setIsTranslatingMsg(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setDisplayContent(raw);
          setIsTranslatingMsg(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [msg.content, raw, selectedLanguageCode, translationCache]);

  if (isGlobalTranslating || isTranslatingMsg) {
    return <UserMessageSkeleton isLatestUser={isLatestUser} latestTurnRef={latestTurnRef} />;
  }

  return (
    <div
      ref={isLatestUser ? latestTurnRef : null}
      className="flex items-start justify-end gap-2 sm:gap-3 group pt-1 scroll-mt-4"
    >
      <div className="flex flex-col items-end max-w-[90%] sm:max-w-lg">
        <div className="bg-white text-slate-800 border border-slate-200/80 rounded-2xl sm:rounded-3xl px-3.5 sm:px-5 py-2.5 sm:py-3 text-xs sm:text-sm shadow-sm font-normal break-words leading-relaxed">
          {displayContent || raw}
        </div>
        <div className="flex items-center gap-3 mt-1.5 text-[11px] text-slate-400">
          <button
            onClick={() => setInputText(displayContent || msg.content)}
            className="flex items-center gap-1 hover:text-slate-600 transition"
          >
            <Pencil1Icon className="w-3 h-3" />
            <span>{uiText?.edit || "Edit"}</span>
          </button>
          <button
            onClick={() => handleSendMessage(msg.originalContent || msg.content)}
            className="flex items-center gap-1 hover:text-slate-600 transition"
          >
            <ReloadIcon className="w-3 h-3" />
            <span>{uiText?.resend || "Resend"}</span>
          </button>
        </div>
      </div>
      <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-gradient-to-tr from-emerald-600 to-green-600 text-white font-semibold text-xs flex items-center justify-center shrink-0 shadow-2xs">
        S
      </div>
    </div>
  );
}

// Dedicated reactive component for assistant answer text with automatic language translation
function AssistantMessageText({ msg, selectedLanguageCode, translationCache }) {
  const raw = msg.originalAnswer || msg.answer || "";
  const [displayText, setDisplayText] = useState(msg.answer || raw);

  useEffect(() => {
    let cancelled = false;
    if (!selectedLanguageCode || selectedLanguageCode === "en") {
      setDisplayText(raw);
      return;
    }
    if (msg.displayLanguageCode === selectedLanguageCode && msg.answer && msg.answer !== raw) {
      setDisplayText(msg.answer);
      return;
    }
    const cacheKey = `en->${selectedLanguageCode}::${raw.trim()}`;
    const cache = translationCache?.current;
    if (cache && cache.has(cacheKey)) {
      setDisplayText(cache.get(cacheKey));
      return;
    }

    translateBatch([raw], selectedLanguageCode, "en", cache).then(([translated]) => {
      if (!cancelled && translated && translated !== raw) {
        setDisplayText(translated);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [msg.answer, raw, msg.displayLanguageCode, selectedLanguageCode, translationCache]);

  // Each [n] is drawn like the source card's [n] badge and jumps to it.
  const parts = String(displayText || "").split(/(\[\d+\])/g);
  return (
    <p className="text-xs sm:text-[15px] leading-relaxed text-slate-800 font-normal break-words">
      {parts.map((part, i) => {
        const match = /^\[(\d+)\]$/.exec(part);
        if (!match) return <React.Fragment key={i}>{part}</React.Fragment>;
        return (
          <button
            key={i}
            type="button"
            onClick={() =>
              document
                .getElementById(`src-${msg.id}-${match[1]}`)
                ?.scrollIntoView({ behavior: "smooth", block: "center" })
            }
            className="mx-0.5 inline-flex items-center rounded-md bg-emerald-600 px-1 text-[10px] sm:text-[11px] font-bold leading-4 text-white align-baseline hover:bg-emerald-700"
          >
            [{match[1]}]
          </button>
        );
      })}
    </p>
  );
}

export default function Home() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    if (typeof window !== "undefined") {
      setSidebarOpen(window.innerWidth >= 1024);
    }
  }, []);
  const [activeNavItem, setActiveNavItem] = useState("chat");
  const [chatSessions, setChatSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);

  // Active chat messages
  const [messages, setMessages] = useState([]);

  const [inputText, setInputText] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [showGuide, setShowGuide] = useState(false);
  const [guideCompleted, setGuideCompleted] = useState(false);
  const [placeholderQuestionIndex, setPlaceholderQuestionIndex] = useState(0);
  const [selectedLanguageCode, setSelectedLanguageCode] = useState("en");
  const [availableVoices, setAvailableVoices] = useState([]);
  const [selectedVoiceURI, setSelectedVoiceURI] = useState("");
  const [recognitionSupported, setRecognitionSupported] = useState(true);
  const [speechSupported, setSpeechSupported] = useState(true);
  const [isListening, setIsListening] = useState(false);
  const [isSpeakingMessageId, setIsSpeakingMessageId] = useState(null);
  const [interimTranscript, setInterimTranscript] = useState("");
  const [voiceMessage, setVoiceMessage] = useState("");
  const [isTranslating, setIsTranslating] = useState(false);
  const [translationNotice, setTranslationNotice] = useState("");
  const [health, setHealth] = useState(null);

  const stepTimerRef = useRef(null);
  const abortControllerRef = useRef(null);
  const latestTurnRef = useRef(null);
  const textareaRef = useRef(null);
  const recognitionRef = useRef(null);
  const translationCacheRef = useRef(new Map());

  const uiText = getUIText(selectedLanguageCode);
  const selectedLanguage = languageByCode(selectedLanguageCode);
  const matchingVoices = voicesForLocale(
    availableVoices,
    selectedLanguage.speechLocale,
  );
  const activeQuestions = DEMO_QUESTIONS;

  // 1.5 second delay for Guide popup on startup
  useEffect(() => {
    const timer = setTimeout(() => {
      setShowGuide(true);
    }, 1500);
    return () => clearTimeout(timer);
  }, []);

  // Persist chat sessions in localStorage (cleaning up any legacy dummy sessions)
  useEffect(() => {
    try {
      const saved = localStorage.getItem("ipsakti_chat_sessions");
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed)) {
          const cleaned = parsed.filter((s) => s.id !== "session-init-1");
          setChatSessions(cleaned);
        }
      }
    } catch { }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("ipsakti_chat_sessions", JSON.stringify(chatSessions));
    } catch { }
  }, [chatSessions]);

  // Requirement 1: Dynamically translate active messages and recent chat titles when language changes with Skeleton Loader
  useEffect(() => {
    let cancelled = false;

    async function handleLanguageSwitch() {
      // 1. If English, instantaneously restore original texts with no skeleton delay
      if (selectedLanguageCode === "en") {
        setIsTranslating(false);
        setMessages((prev) =>
          prev.map((m) => {
            if (m.role === "user") {
              return {
                ...m,
                content: m.originalContent || m.content,
              };
            }
            if (m.role === "assistant") {
              return {
                ...m,
                answer: m.originalAnswer || m.answer,
                caution: m.caution
                  ? { ...m.caution, detail: m.originalCautionDetail || m.caution.detail }
                  : m.caution,
                disclaimer: m.originalDisclaimer || m.disclaimer,
                displayLanguageCode: "en",
                evidence: (m.evidence || []).map((src) => ({
                  ...src,
                  document_name: src.originalDocumentName || src.document_name,
                })),
              };
            }
            return m;
          }),
        );
        setChatSessions((prev) =>
          prev.map((s) => ({
            ...s,
            title: s.originalTitle || s.title,
            messages: (s.messages || []).map((m) => {
              if (m.role === "user") {
                return {
                  ...m,
                  content: m.originalContent || m.content,
                };
              }
              if (m.role === "assistant") {
                return {
                  ...m,
                  answer: m.originalAnswer || m.answer,
                  caution: m.caution
                    ? { ...m.caution, detail: m.originalCautionDetail || m.caution.detail }
                    : m.caution,
                  disclaimer: m.originalDisclaimer || m.disclaimer,
                  displayLanguageCode: "en",
                  evidence: (m.evidence || []).map((src) => ({
                    ...src,
                    document_name: src.originalDocumentName || src.document_name,
                  })),
                };
              }
              return m;
            }),
          })),
        );
        return;
      }

      // 2. If non-English, show skeleton loader until all texts are fully translated
      const hasContentToTranslate =
        messages.some((m) => m.role === "assistant" || m.role === "user") ||
        chatSessions.some((s) => s.title);

      if (hasContentToTranslate) {
        setIsTranslating(true);
      }

      try {
        const cache = translationCacheRef.current;
        const lang = selectedLanguageCode;

        // ── STEP 1: Collect all unique texts to translate in ONE batch ─────
        const jobs = [];

        // Active messages
        messages.forEach((m, mi) => {
          if (m.role === "user") {
            const baseContent = m.originalContent || m.content;
            if (baseContent) jobs.push({ type: "userMsg", mi, text: baseContent });
            return;
          }
          if (m.role !== "assistant") return;
          const baseAnswer = m.originalAnswer || m.answer;
          const baseCaution = m.originalCautionDetail || m.caution?.detail;
          const baseDisclaimer = m.originalDisclaimer || m.disclaimer;
          if (baseAnswer) jobs.push({ type: "msg", mi, field: "answer", text: baseAnswer });
          if (baseCaution) jobs.push({ type: "msg", mi, field: "caution", text: baseCaution });
          if (baseDisclaimer) jobs.push({ type: "msg", mi, field: "disclaimer", text: baseDisclaimer });
          (m.evidence || []).forEach((src, sIdx) => {
            const baseDocName = src.originalDocumentName || src.document_name;
            if (baseDocName) jobs.push({ type: "evidenceDoc", mi, sIdx, text: baseDocName });
          });
        });

        // Session titles + session message fields
        chatSessions.forEach((s, si) => {
          const baseTitle = s.originalTitle || s.title;
          if (baseTitle) jobs.push({ type: "session", si, field: "title", text: baseTitle });
          (s.messages || []).forEach((m, mi) => {
            if (m.role === "user") {
              const baseContent = m.originalContent || m.content;
              if (baseContent) jobs.push({ type: "sessionUserMsg", si, mi, text: baseContent });
              return;
            }
            if (m.role !== "assistant") return;
            const baseAnswer = m.originalAnswer || m.answer;
            const baseCaution = m.originalCautionDetail || m.caution?.detail;
            const baseDisclaimer = m.originalDisclaimer || m.disclaimer;
            if (baseAnswer) jobs.push({ type: "sessionMsg", si, mi, field: "answer", text: baseAnswer });
            if (baseCaution) jobs.push({ type: "sessionMsg", si, mi, field: "caution", text: baseCaution });
            if (baseDisclaimer) jobs.push({ type: "sessionMsg", si, mi, field: "disclaimer", text: baseDisclaimer });
            (m.evidence || []).forEach((src, sIdx) => {
              const baseDocName = src.originalDocumentName || src.document_name;
              if (baseDocName) jobs.push({ type: "sessionEvidenceDoc", si, mi, sIdx, text: baseDocName });
            });
          });
        });

        if (jobs.length === 0) return;

        // ── STEP 2: Single batch call for all texts ────────────────────────
        const allTexts = jobs.map((j) => j.text);
        const allTranslated = await translateBatch(allTexts, lang, "en", cache);

        if (cancelled) return;

        // ── STEP 3: Apply results ─────────────────────────────────────────
        const msgUpdates = {};
        const userMsgUpdates = {};
        const evidenceDocUpdates = {};
        const sessionTitleUpdates = {};
        const sessionMsgUpdates = {};
        const sessionUserMsgUpdates = {};
        const sessionEvidenceDocUpdates = {};

        jobs.forEach((job, jIdx) => {
          const translated = allTranslated[jIdx] || job.text;
          if (job.type === "msg") {
            if (!msgUpdates[job.mi]) msgUpdates[job.mi] = {};
            msgUpdates[job.mi][job.field] = translated;
            msgUpdates[job.mi][`orig_${job.field}`] = job.text;
          } else if (job.type === "userMsg") {
            userMsgUpdates[job.mi] = { content: translated, origContent: job.text };
          } else if (job.type === "evidenceDoc") {
            evidenceDocUpdates[`${job.mi}-${job.sIdx}`] = { translated, original: job.text };
          } else if (job.type === "session") {
            sessionTitleUpdates[job.si] = { translated, original: job.text };
          } else if (job.type === "sessionMsg") {
            const key = `${job.si}-${job.mi}`;
            if (!sessionMsgUpdates[key]) sessionMsgUpdates[key] = {};
            sessionMsgUpdates[key][job.field] = translated;
            sessionMsgUpdates[key][`orig_${job.field}`] = job.text;
          } else if (job.type === "sessionUserMsg") {
            sessionUserMsgUpdates[`${job.si}-${job.mi}`] = { content: translated, origContent: job.text };
          } else if (job.type === "sessionEvidenceDoc") {
            sessionEvidenceDocUpdates[`${job.si}-${job.mi}-${job.sIdx}`] = { translated, original: job.text };
          }
        });

        const updatedMessages = messages.map((m, mi) => {
          if (m.role === "user") {
            const userUpd = userMsgUpdates[mi];
            if (!userUpd) return m;
            return {
              ...m,
              originalContent: userUpd.origContent ?? (m.originalContent || m.content),
              content: userUpd.content || m.content,
            };
          }
          if (m.role !== "assistant") return m;
          const upd = msgUpdates[mi] || {};
          const updatedEvidence = (m.evidence || []).map((src, sIdx) => {
            const docUpd = evidenceDocUpdates[`${mi}-${sIdx}`];
            return {
              ...src,
              originalDocumentName: docUpd?.original ?? (src.originalDocumentName || src.document_name),
              document_name: docUpd?.translated || src.document_name,
            };
          });
          return {
            ...m,
            originalAnswer: upd.orig_answer ?? (m.originalAnswer || m.answer),
            originalCautionDetail: upd.orig_caution ?? (m.originalCautionDetail || m.caution?.detail),
            originalDisclaimer: upd.orig_disclaimer ?? (m.originalDisclaimer || m.disclaimer),
            answer: upd.answer || m.answer,
            caution: m.caution
              ? { ...m.caution, detail: upd.caution || m.caution.detail }
              : m.caution,
            disclaimer: upd.disclaimer || m.disclaimer,
            displayLanguageCode: lang,
            evidence: updatedEvidence,
          };
        });

        const updatedSessions = chatSessions.map((s, si) => {
          const titleUpd = sessionTitleUpdates[si];
          const updMsgs = (s.messages || []).map((m, mi) => {
            if (m.role === "user") {
              const userUpd = sessionUserMsgUpdates[`${si}-${mi}`];
              if (!userUpd) return m;
              return {
                ...m,
                originalContent: userUpd.origContent ?? (m.originalContent || m.content),
                content: userUpd.content || m.content,
              };
            }
            if (m.role !== "assistant") return m;
            const key = `${si}-${mi}`;
            const upd = sessionMsgUpdates[key] || {};
            const updatedEvidence = (m.evidence || []).map((src, sIdx) => {
              const docUpd = sessionEvidenceDocUpdates[`${si}-${mi}-${sIdx}`];
              return {
                ...src,
                originalDocumentName: docUpd?.original ?? (src.originalDocumentName || src.document_name),
                document_name: docUpd?.translated || src.document_name,
              };
            });
            return {
              ...m,
              originalAnswer: upd.orig_answer ?? (m.originalAnswer || m.answer),
              originalCautionDetail: upd.orig_caution ?? (m.originalCautionDetail || m.caution?.detail),
              originalDisclaimer: upd.orig_disclaimer ?? (m.originalDisclaimer || m.disclaimer),
              answer: upd.answer || m.answer,
              caution: m.caution
                ? { ...m.caution, detail: upd.caution || m.caution.detail }
                : m.caution,
              disclaimer: upd.disclaimer || m.disclaimer,
              displayLanguageCode: lang,
              evidence: updatedEvidence,
            };
          });
          return {
            ...s,
            originalTitle: titleUpd?.original ?? (s.originalTitle || s.title),
            title: titleUpd?.translated || s.title,
            messages: updMsgs,
          };
        });

        if (!cancelled) {
          setMessages(updatedMessages);
          setChatSessions(updatedSessions);
        }
      } catch {
        // Fallback: keep current texts
      } finally {
        if (!cancelled) {
          setIsTranslating(false);
        }
      }
    }

    handleLanguageSwitch();

    return () => {
      cancelled = true;
    };
  }, [selectedLanguageCode]);

  useEffect(() => {
    fetchHealth();
    const synth = window.speechSynthesis;
    setSpeechSupported(
      "speechSynthesis" in window && "SpeechSynthesisUtterance" in window,
    );
    const loadVoices = () => setAvailableVoices(synth?.getVoices() || []);
    loadVoices();
    synth?.addEventListener?.("voiceschanged", loadVoices);
    return () => {
      synth?.removeEventListener?.("voiceschanged", loadVoices);
      synth?.cancel();
    };
  }, []);

  async function fetchHealth() {
    for (const apiUrl of [PRIMARY_API_URL, FALLBACK_API_URL]) {
      try {
        const response = await fetch(`${apiUrl}/api/health`);
        if (response.ok) {
          setHealth(await response.json());
          return;
        }
      } catch {
        // Fallback
      }
    }
    setHealth(null);
  }

  useEffect(() => {
    const preferredVoice = selectPreferredVoice(
      availableVoices,
      selectedLanguage.speechLocale,
      selectedVoiceURI,
    );
    setSelectedVoiceURI(preferredVoice?.voiceURI || "");
  }, [availableVoices, selectedLanguage.speechLocale, selectedVoiceURI]);

  useEffect(() => {
    const Recognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) {
      setRecognitionSupported(false);
      recognitionRef.current = null;
      return undefined;
    }

    setRecognitionSupported(true);
    const recognition = new Recognition();
    recognition.lang = selectedLanguage.speechLocale;
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.onresult = (event) => {
      let finalText = "";
      let interimText = "";
      for (let index = 0; index < event.results.length; index += 1) {
        const transcript = event.results[index][0]?.transcript || "";
        if (event.results[index].isFinal) finalText += transcript;
        else interimText += transcript;
      }
      if (finalText.trim()) setInputText(finalText.trim());
      setInterimTranscript(interimText.trim());
    };
    recognition.onerror = (event) => {
      setIsListening(false);
      setInterimTranscript("");
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        setVoiceMessage(
          "Microphone access was denied. Allow access in your browser or type your question.",
        );
      } else if (event.error !== "aborted") {
        setVoiceMessage(
          "Voice input stopped. You can try again or type your question.",
        );
      }
    };
    recognition.onend = () => {
      setIsListening(false);
      setInterimTranscript("");
    };
    recognitionRef.current = recognition;
    return () => {
      recognition.onend = null;
      recognition.abort();
      if (recognitionRef.current === recognition) recognitionRef.current = null;
    };
  }, [selectedLanguage.speechLocale]);

  // Cycle through demo questions one-by-one when textarea is empty
  useEffect(() => {
    if (inputText.trim()) return;
    const interval = setInterval(() => {
      setPlaceholderQuestionIndex(
        (prev) => (prev + 1) % (activeQuestions.length || 1),
      );
    }, 4500);
    return () => clearInterval(interval);
  }, [inputText, activeQuestions.length]);

  // Auto-fill active placeholder question on Tab key or placeholder click
  const applyPlaceholderText = () => {
    if (!inputText) {
      const activeQuestion = activeQuestions[placeholderQuestionIndex];
      if (activeQuestion) {
        setInputText(activeQuestion);
        setTimeout(() => {
          if (textareaRef.current) {
            textareaRef.current.focus();
            try {
              textareaRef.current.setSelectionRange(
                activeQuestion.length,
                activeQuestion.length,
              );
            } catch (err) { }
          }
        }, 0);
      }
    }
  };

  const handleInputKeyDown = (e) => {
    if (e.key === "Tab" && !inputText) {
      e.preventDefault();
      applyPlaceholderText();
    }
  };

  useEffect(() => {
    const handleGlobalKeyDown = (e) => {
      if (e.key === "Tab" && !inputText && messages.length === 0 && !showGuide) {
        e.preventDefault();
        applyPlaceholderText();
      }
    };
    window.addEventListener("keydown", handleGlobalKeyDown);
    return () => window.removeEventListener("keydown", handleGlobalKeyDown);
  }, [inputText, messages.length, showGuide, placeholderQuestionIndex, activeQuestions]);

  useEffect(() => {
    if (latestTurnRef.current) {
      latestTurnRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [messages.length, isGenerating]);

  const clearStepTimers = () => {
    if (Array.isArray(stepTimerRef.current)) {
      stepTimerRef.current.forEach(clearTimeout);
      stepTimerRef.current = [];
    } else if (stepTimerRef.current) {
      clearInterval(stepTimerRef.current);
      stepTimerRef.current = null;
    }
  };

  const handleStartNewChat = () => {
    stopGeneration();
    recognitionRef.current?.abort();
    window.speechSynthesis?.cancel();
    setMessages([]);
    setInputText("");
    setCurrentStepIndex(0);
    setActiveSessionId(null);
    setIsListening(false);
    setIsSpeakingMessageId(null);
    setInterimTranscript("");
    setTranslationNotice("");
    setVoiceMessage("");
  };

  const handleSelectSession = async (sessionId) => {
    const session = chatSessions.find((s) => s.id === sessionId);
    if (!session) return;
    setActiveSessionId(sessionId);
    const sessionMsgs = session.messages || [];
    setInputText("");
    if (typeof window !== "undefined" && window.innerWidth < 1024) {
      setSidebarOpen(false);
    }

    if (selectedLanguageCode === "en") {
      setMessages(
        sessionMsgs.map((m) => {
          if (m.role === "user") {
            return {
              ...m,
              content: m.originalContent || m.content,
            };
          }
          if (m.role === "assistant") {
            return {
              ...m,
              answer: m.originalAnswer || m.answer,
              caution: m.caution
                ? { ...m.caution, detail: m.originalCautionDetail || m.caution.detail }
                : m.caution,
              disclaimer: m.originalDisclaimer || m.disclaimer,
              displayLanguageCode: "en",
              evidence: (m.evidence || []).map((src) => ({
                ...src,
                document_name: src.originalDocumentName || src.document_name,
              })),
            };
          }
          return m;
        }),
      );
      return;
    }

    const needsTranslation = sessionMsgs.some(
      (m) =>
        (m.role === "assistant" && m.displayLanguageCode !== selectedLanguageCode) ||
        (m.role === "user" && m.originalContent),
    );

    if (needsTranslation) {
      setIsTranslating(true);
      setMessages(sessionMsgs);
      try {
        const cache = translationCacheRef.current;
        const lang = selectedLanguageCode;

        const jobs = [];
        sessionMsgs.forEach((m, mi) => {
          if (m.role === "user") {
            const baseContent = m.originalContent || m.content;
            if (baseContent) jobs.push({ type: "user", mi, text: baseContent });
            return;
          }
          if (m.role !== "assistant") return;
          const baseAns = m.originalAnswer || m.answer;
          const baseC = m.originalCautionDetail || m.caution?.detail;
          const baseD = m.originalDisclaimer || m.disclaimer;
          if (baseAns) jobs.push({ type: "assistant", mi, field: "answer", text: baseAns });
          if (baseC) jobs.push({ type: "assistant", mi, field: "caution", text: baseC });
          if (baseD) jobs.push({ type: "assistant", mi, field: "disclaimer", text: baseD });
          (m.evidence || []).forEach((src, sIdx) => {
            const baseDocName = src.originalDocumentName || src.document_name;
            if (baseDocName) jobs.push({ type: "evidenceDoc", mi, sIdx, text: baseDocName });
          });
        });

        const allTexts = jobs.map((j) => j.text);
        const allTranslated = await translateBatch(allTexts, lang, "en", cache);

        const updates = {};
        const userUpdates = {};
        const docUpdates = {};

        jobs.forEach((job, idx) => {
          const translated = allTranslated[idx] || job.text;
          if (job.type === "assistant") {
            if (!updates[job.mi]) updates[job.mi] = {};
            updates[job.mi][job.field] = translated;
            updates[job.mi][`orig_${job.field}`] = job.text;
          } else if (job.type === "user") {
            userUpdates[job.mi] = { content: translated, origContent: job.text };
          } else if (job.type === "evidenceDoc") {
            docUpdates[`${job.mi}-${job.sIdx}`] = { translated, original: job.text };
          }
        });

        const translatedMsgs = sessionMsgs.map((m, mi) => {
          if (m.role === "user") {
            const uUpd = userUpdates[mi];
            if (!uUpd) return m;
            return {
              ...m,
              originalContent: uUpd.origContent ?? (m.originalContent || m.content),
              content: uUpd.content || m.content,
            };
          }
          if (m.role !== "assistant") return m;
          const upd = updates[mi] || {};
          const updatedEvidence = (m.evidence || []).map((src, sIdx) => {
            const dUpd = docUpdates[`${mi}-${sIdx}`];
            return {
              ...src,
              originalDocumentName: dUpd?.original ?? (src.originalDocumentName || src.document_name),
              document_name: dUpd?.translated || src.document_name,
            };
          });
          return {
            ...m,
            originalAnswer: upd.orig_answer ?? (m.originalAnswer || m.answer),
            originalCautionDetail: upd.orig_caution ?? (m.originalCautionDetail || m.caution?.detail),
            originalDisclaimer: upd.orig_disclaimer ?? (m.originalDisclaimer || m.disclaimer),
            answer: upd.answer || m.answer,
            caution: m.caution ? { ...m.caution, detail: upd.caution || m.caution.detail } : m.caution,
            disclaimer: upd.disclaimer || m.disclaimer,
            displayLanguageCode: lang,
            evidence: updatedEvidence,
          };
        });
        setMessages(translatedMsgs);
      } finally {
        setIsTranslating(false);
      }
    } else {
      setMessages(sessionMsgs);
    }
  };

  const handleDeleteSession = (sessionId) => {
    setChatSessions((prev) => prev.filter((s) => s.id !== sessionId));
    if (activeSessionId === sessionId) {
      handleStartNewChat();
    }
  };

  const handleClearConversation = () => {
    if (activeSessionId) {
      handleDeleteSession(activeSessionId);
    } else {
      handleStartNewChat();
    }
  };

  const stopGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    clearStepTimers();
    setIsGenerating(false);
  };

  const toggleListening = () => {
    const recognition = recognitionRef.current;
    if (!recognition || isGenerating) return;
    setVoiceMessage("");
    if (isListening) {
      recognition.stop();
      setIsListening(false);
      return;
    }

    setInterimTranscript("");
    try {
      recognition.start();
      setIsListening(true);
    } catch {
      setVoiceMessage(
        "Voice input is already active. Stop it before starting again.",
      );
    }
  };

  const toggleSpeech = (message) => {
    if (!speechSupported || !message?.answer) return;
    if (isSpeakingMessageId === message.id) {
      window.speechSynthesis.cancel();
      setIsSpeakingMessageId(null);
      return;
    }

    const answerLanguage = languageByCode(message.displayLanguageCode || "en");
    const preferredVoiceURI =
      answerLanguage.code === selectedLanguage.code ? selectedVoiceURI : "";
    const voice = selectPreferredVoice(
      availableVoices,
      answerLanguage.speechLocale,
      preferredVoiceURI,
    );
    const utterance = new SpeechSynthesisUtterance(message.answer);
    utterance.lang = voice?.lang || answerLanguage.speechLocale;
    if (voice) utterance.voice = voice;
    utterance.onend = () => setIsSpeakingMessageId(null);
    utterance.onerror = () => {
      setIsSpeakingMessageId(null);
      setVoiceMessage("Audio playback is unavailable in this browser.");
    };
    window.speechSynthesis.cancel();
    setIsSpeakingMessageId(message.id);
    window.speechSynthesis.speak(utterance);
  };

  const handleSendMessage = async (queryToSend) => {
    const q = (queryToSend || inputText).trim();
    if (!q || isGenerating) return;

    let currentSessionId = activeSessionId;
    if (!currentSessionId) {
      currentSessionId = `session-${Date.now()}`;
      setActiveSessionId(currentSessionId);
      const newSession = {
        id: currentSessionId,
        title: q.slice(0, 32).trim() + (q.length > 32 ? "..." : ""),
        originalTitle: q.slice(0, 32).trim() + (q.length > 32 ? "..." : ""),
        messages: [],
        timestamp: Date.now(),
      };
      setChatSessions((prev) => [newSession, ...prev]);

      // Call backend Groq AI to generate concise 3-5 word chat title
      fetch("/api/chat-title", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          language: selectedLanguage.label,
          languageCode: selectedLanguageCode,
        }),
      })
        .then((res) => res.json())
        .then((data) => {
          if (data?.title) {
            setChatSessions((prev) =>
              prev.map((s) =>
                s.id === currentSessionId
                  ? {
                      ...s,
                      title: data.title,
                      originalTitle:
                        selectedLanguageCode === "en"
                          ? data.title
                          : s.originalTitle || data.title,
                    }
                  : s,
              ),
            );
          }
        })
        .catch(() => { });
    }

    const requestLanguage = selectedLanguage;

    setInputText("");
    const userMsgId = `user-${Date.now()}`;
    const aiMsgId = `ai-${Date.now()}`;

    const userMessage = {
      id: userMsgId,
      role: "user",
      content: q,
      originalContent: q,
    };

    const aiPlaceholder = {
      id: aiMsgId,
      role: "assistant",
      status: "loading",
      step: 0,
    };

    setMessages((prev) => [...prev, userMessage, aiPlaceholder]);
    setChatSessions((prev) =>
      prev.map((s) =>
        s.id === currentSessionId
          ? { ...s, messages: [...(s.messages || []), userMessage, aiPlaceholder] }
          : s,
      ),
    );

    setIsGenerating(true);
    setCurrentStepIndex(0);
    setTranslationNotice("");

    clearStepTimers();
    const stepDelays = [300, 800, 600, 450, 1200];
    const timeouts = [];
    let accumulated = 0;

    stepDelays.forEach((delay, idx) => {
      accumulated += delay;
      const t = setTimeout(() => {
        setCurrentStepIndex(idx + 1);
      }, accumulated);
      timeouts.push(t);
    });

    stepTimerRef.current = timeouts;

    abortControllerRef.current = new AbortController();

    try {
      let rawData = null;
      let englishQuery = q;

      // Translate from the language the query is written in, not only the
      // dropdown's: Hindi typed under an English dropdown must still reach the
      // English-only retriever as English. The answer language stays the
      // dropdown's (`requestLanguage`).
      const queryLanguageCode = resolveQueryLanguage(q, requestLanguage.code);
      if (queryLanguageCode !== "en") {
        try {
          englishQuery = await translateText({
            text: q,
            sourceLanguage: queryLanguageCode,
            targetLanguage: "en",
            cache: translationCacheRef.current,
          });
          if (queryTranslationFailed(q, englishQuery)) {
            throw new Error(TRANSLATION_FALLBACK);
          }
          if (queryLanguageCode !== requestLanguage.code) {
            const typed = languageByCode(queryLanguageCode).label;
            setTranslationNotice(
              `Your ${typed} question was translated for search. Answers follow the selected language (${requestLanguage.label}); choose ${typed} in the language menu to read answers in ${typed}.`,
            );
          }
        } catch (translationError) {
          clearStepTimers();
          setTranslationNotice(
            translationError.message || TRANSLATION_FALLBACK,
          );
          setMessages((prev) =>
            prev.map((message) =>
              message.id === aiMsgId
                ? {
                  ...message,
                  status: "error",
                  error:
                    "Translation could not start this request. Please continue in English.",
                }
                : message,
            ),
          );
          return;
        }
      }

      let serverErrorMessage = null;
      try {
        const queryToSend = (englishQuery || q).trim();
        const response = await fetch(`${PRIMARY_API_URL}/api/ask`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: queryToSend }),
          signal: abortControllerRef.current.signal,
        });
        if (response.ok) {
          rawData = await response.json();
        } else {
          const errData = await response.json().catch(() => ({}));
          serverErrorMessage = errData.message || `RAG service returned status ${response.status}`;
        }
      } catch {
        try {
          const responseFb = await fetch(`${FALLBACK_API_URL}/api/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: (englishQuery || q).trim() }),
            signal: abortControllerRef.current.signal,
          });
          if (responseFb.ok) {
            rawData = await responseFb.json();
          } else {
            const errDataFb = await responseFb.json().catch(() => ({}));
            serverErrorMessage = errDataFb.message || `Fallback API returned status ${responseFb.status}`;
          }
        } catch {
          // Fallback also failed
        }
      }

      clearStepTimers();
      setCurrentStepIndex(ANALYSIS_STEPS.length);
      await new Promise((resolve) => setTimeout(resolve, 350));

      if (rawData) {
        const answerObj = rawData.answer || {};
        // SOCIAL, NEEDS_CLARIFICATION, UNAVAILABLE and ABSTAINED each have
        // their own text; only ANSWERED carries confidence and citations.
        const view = describeResult(rawData);
        const isAnswered = view.kind === "answered";
        const originalAnswerText = view.text || "No answer was published.";
        let answerText = originalAnswerText;
        let displayLanguageCode = "en";

        const confidenceVal = isAnswered ? answerObj.confidence || null : null;

        let cautionItem = null;
        if (isAnswered && answerObj.cautions && answerObj.cautions.length > 0) {
          cautionItem = {
            title: "Source notice",
            detail: answerObj.cautions[0],
          };
        }
        if (view.sources === "none") {
          rawData.evidence = [];
        }

        const originalCautionDetail = cautionItem?.detail || null;
        const originalDisclaimer = rawData.disclaimer || null;
        let translatedDisclaimer = rawData.disclaimer || null;

        // Normalize evidence to store originalDocumentName and originalText
        if (Array.isArray(rawData.evidence)) {
          rawData.evidence = rawData.evidence.map((src) => ({
            ...src,
            originalDocumentName: src.originalDocumentName || src.document_name,
            originalText: src.originalText || src.text,
          }));
        }

        let finalUserContent = q;

        if (requestLanguage.code !== "en") {
          try {
            // Translate answer, caution, disclaimer, user question, and evidence doc names in ONE batch call
            const docNameTexts = (rawData.evidence || []).map((src) => src.originalDocumentName || src.document_name || "");
            const textsToTranslate = [
              originalAnswerText || "",
              cautionItem?.detail || "",
              rawData.disclaimer || "",
              q,
              ...docNameTexts,
            ];
            const translated = await translateBatch(
              textsToTranslate,
              requestLanguage.code,
              "en",
              translationCacheRef.current,
            );
            if (translated[0] && originalAnswerText && translated[0].trim() !== originalAnswerText.trim()) {
              answerText = translated[0];
              displayLanguageCode = requestLanguage.code;
            }
            if (translated[1] && cautionItem?.detail) {
              cautionItem.detail = translated[1];
            }
            if (translated[2] && rawData.disclaimer) {
              translatedDisclaimer = translated[2];
            }
            if (translated[3] && translated[3].trim() !== q.trim()) {
              finalUserContent = translated[3];
            }
            if (Array.isArray(rawData.evidence)) {
              rawData.evidence = rawData.evidence.map((src, idx) => ({
                ...src,
                originalDocumentName: src.originalDocumentName || src.document_name,
                document_name: translated[4 + idx] || src.document_name,
              }));
            }
          } catch (translationError) {
            setTranslationNotice(
              translationError.message || TRANSLATION_FALLBACK,
            );
          }
        }

        const completedAiMessage = {
          id: aiMsgId,
          role: "assistant",
          status: "completed",
          kind: view.kind,
          clarificationOptions: view.options || [],
          confidence: confidenceVal,
          originalAnswer: originalAnswerText,
          originalCautionDetail,
          originalDisclaimer,
          answer: answerText,
          displayLanguageCode,
          jurisdiction: rawData.jurisdiction ?? answerObj.jurisdiction,
          disclaimer: translatedDisclaimer,
          citations: answerObj.citations || [],
          caution: cautionItem,
          evidence: rawData.evidence || [],
          timings_ms: rawData.timings_ms || {},
        };

        setMessages((prev) =>
          prev.map((m) => {
            if (m.id === userMsgId && finalUserContent !== q) {
              return { ...m, content: finalUserContent, originalContent: q };
            }
            if (m.id === aiMsgId) {
              return completedAiMessage;
            }
            return m;
          }),
        );
        setChatSessions((prev) =>
          prev.map((s) => {
            if (s.id !== currentSessionId) return s;
            const currentMsgs = (s.messages || []).map((m) => {
              if (m.id === userMsgId && finalUserContent !== q) {
                return { ...m, content: finalUserContent, originalContent: q };
              }
              if (m.id === aiMsgId) {
                return completedAiMessage;
              }
              return m;
            });
            const hasAiMsg = currentMsgs.some((m) => m.id === aiMsgId);
            return {
              ...s,
              messages: hasAiMsg ? currentMsgs : [...currentMsgs, completedAiMessage],
            };
          }),
        );
      } else {
        const errorMsg = {
          id: aiMsgId,
          role: "assistant",
          status: "error",
          error:
            serverErrorMessage ||
            "IP-SAKTI could not complete that request. Check that the RAG service is running and try again.",
        };
        setMessages((prev) =>
          prev.map((message) => (message.id === aiMsgId ? errorMsg : message)),
        );
        setChatSessions((prev) =>
          prev.map((s) => {
            if (s.id !== currentSessionId) return s;
            const currentMsgs = s.messages || [];
            const hasAiMsg = currentMsgs.some((m) => m.id === aiMsgId);
            return {
              ...s,
              messages: hasAiMsg
                ? currentMsgs.map((m) => (m.id === aiMsgId ? errorMsg : m))
                : [...currentMsgs, errorMsg],
            };
          }),
        );
      }
    } catch (err) {
      clearStepTimers();
      if (err.name === "AbortError") {
        setMessages((prev) => prev.filter((m) => m.id !== aiMsgId));
        setChatSessions((prev) =>
          prev.map((s) =>
            s.id === currentSessionId
              ? {
                ...s,
                messages: (s.messages || []).filter((m) => m.id !== aiMsgId),
              }
              : s,
          ),
        );
      } else {
        const errorMsg = {
          id: aiMsgId,
          role: "assistant",
          status: "error",
          error:
            err?.message
              ? `IP-SAKTI encountered an issue: ${err.message}`
              : "IP-SAKTI could not complete that request. Check that the RAG service is running and try again.",
        };
        setMessages((prev) =>
          prev.map((message) => (message.id === aiMsgId ? errorMsg : message)),
        );
        setChatSessions((prev) =>
          prev.map((s) => {
            if (s.id !== currentSessionId) return s;
            const currentMsgs = s.messages || [];
            const hasAiMsg = currentMsgs.some((m) => m.id === aiMsgId);
            return {
              ...s,
              messages: hasAiMsg
                ? currentMsgs.map((m) => (m.id === aiMsgId ? errorMsg : m))
                : [...currentMsgs, errorMsg],
            };
          }),
        );
      }
    } finally {
      setIsGenerating(false);
    }
  };

  const setQuestion = setInputText;
  const handleAsk = handleSendMessage;

  return (
    <div className="flex h-screen h-[100dvh] w-screen overflow-hidden text-[#0f172a] relative bg-[#f8faf9]">
      {/* ─── SHADCN.IO ANIMATED GREEN GRADIENT BACKGROUND (BRIGHT LOBES) ── */}
      <div className="fixed inset-0 pointer-events-none z-0 opacity-15">
        <BackgroundGradientAnimation
          gradientBackgroundStart="rgb(255, 255, 255)"
          gradientBackgroundEnd="rgb(240, 253, 245)"
          firstColor="34, 197, 94"
          secondColor="16, 185, 129"
          thirdColor="74, 222, 128"
          fourthColor="52, 211, 153"
          fifthColor="20, 184, 166"
          pointerColor="34, 197, 94"
          size="90%"
        />
      </div>

      {/* ─── LEFT SIDEBAR (LIGHT THEME & CHAT HISTORY) ───────────────── */}
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        uiText={uiText}
        activeItem={activeNavItem}
        onSelectItem={(id) => {
          setActiveNavItem(id);
          if (typeof window !== "undefined" && window.innerWidth < 1024) {
            setSidebarOpen(false);
          }
        }}
        onOpenTutorial={() => {
          setShowGuide(true);
          if (typeof window !== "undefined" && window.innerWidth < 1024) {
            setSidebarOpen(false);
          }
        }}
        chatSessions={chatSessions}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleStartNewChat}
        onDeleteSession={handleDeleteSession}
        onClearConversation={handleClearConversation}
        isTranslating={isTranslating}
      />

      {/* ─── MAIN CHAT WORKSPACE ────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col h-full min-w-0 relative z-10 overflow-hidden">
        {/* Smooth Animated Sidebar Toggle Button (ChatGPT style) */}
        <AnimatePresence>
          {!sidebarOpen && (
            <motion.button
              key="sidebar-toggle-open-btn"
              initial={{ opacity: 0, scale: 0.85, x: -8 }}
              animate={{ opacity: 1, scale: 1, x: 0 }}
              exit={{ opacity: 0, scale: 0.85, x: -8 }}
              transition={{ duration: 0.22, ease: "easeOut" }}
              type="button"
              onClick={() => setSidebarOpen(true)}
              className="absolute top-3.5 left-3.5 sm:top-4 sm:left-4 z-40 p-2 rounded-xl glass-pill text-slate-600 hover:text-slate-900 hover:bg-white/80 transition-all duration-200 flex items-center justify-center shadow-xs group cursor-pointer active:scale-95"
              title="Open sidebar"
              aria-label="Open sidebar"
            >
              <PanelLeft className="w-5 h-5 text-slate-600 group-hover:text-slate-900 transition-transform duration-200 group-hover:scale-105" />
            </motion.button>
          )}
        </AnimatePresence>

        {/* Voice/Translation Status Notice Banner */}
        {(voiceMessage || translationNotice || interimTranscript || isListening) && (
          <div className="px-3 pt-2 shrink-0">
            <div
              className={`mx-auto max-w-[960px] rounded-xl border px-3 py-1.5 text-[11px] ${voiceMessage || translationNotice
                ? "border-amber-200 bg-amber-50/90 text-amber-800"
                : isListening
                  ? "border-emerald-200 bg-emerald-50/90 text-emerald-700"
                  : "border-slate-200 bg-white/80 text-slate-500"
                }`}
              aria-live="polite"
            >
              {translationNotice ||
                voiceMessage ||
                interimTranscript ||
                (isListening ? `Listening in ${selectedLanguage.label}…` : "")}
            </div>
          </div>
        )}

        {/* ─── MAIN CHAT CANVAS ─────────────────────────────────────────────── */}
        <div className="flex-1 overflow-y-auto px-2.5 sm:px-8 md:px-12 py-4 sm:py-6 space-y-4 sm:space-y-6 scroll-smooth scroll-pt-4 sm:scroll-pt-6 hero-gradient-bg flex flex-col">
          <div className="max-w-[760px] mx-auto space-y-4 sm:space-y-6 w-full flex-1 flex flex-col justify-center">

            {/* ─── HERO EMPTY STATE ─────────────────────────────────────────── */}
            <AnimatePresence mode="wait">
              {messages.length === 0 && !isGenerating && (
                <motion.div
                  key="hero-empty-state"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, transition: { duration: 0.2 } }}
                  className="flex flex-col items-center justify-center flex-1 my-auto min-h-[calc(100dvh-56px-2rem)] sm:min-h-[calc(100dvh-64px-3rem)] text-center px-3 sm:px-2 py-4 sm:py-6"
                >
                  <div className="w-full max-w-[720px] space-y-6 sm:space-y-7">

                    {/* Headline */}
                    <motion.div
                      exit={{ opacity: 0, y: -20, filter: "blur(4px)" }}
                      transition={{ duration: 0.25 }}
                      className="space-y-2 sm:space-y-3"
                    >
                      <h1 className="text-[26px] leading-[1.15] sm:text-4xl md:text-5xl font-normal text-slate-700 tracking-tight">
                        {uiText.namaste},{" "}<span className="font-bold text-[#0f172a]">{uiText.readyToNavigate}</span>
                      </h1>
                      <h2 className="text-[26px] leading-[1.15] sm:text-4xl md:text-5xl font-bold text-[#0f172a] tracking-tight">
                        {uiText.indianIpStatutoryLaw}
                      </h2>
                      <p className="text-[12px] sm:text-sm text-slate-600 max-w-md mx-auto pt-1.5 font-normal px-1">
                        {uiText.statutoryIntelligenceDesc}
                      </p>
                    </motion.div>

                    {/* Main Floating Glass Input Card */}
                    <div className="text-left w-full relative">
                      <motion.div
                        layoutId="unified-chat-card"
                        transition={{ type: "spring", stiffness: 280, damping: 28 }}
                        className="w-full relative z-10"
                      >
                        <ChatInput
                          variant="unstyled"
                          value={inputText}
                          onChange={(e) => setInputText(e.target.value)}
                          onSubmit={() => handleSendMessage()}
                          loading={isGenerating}
                          onStop={stopGeneration}
                          className="glass-card-30 p-3.5 sm:p-5 flex flex-col items-stretch transition-all"
                        >
                          {/* Input Area */}
                          <div className="relative w-full px-0.5 min-h-[48px] sm:min-h-[56px]">
                            <ChatInputTextArea
                              ref={textareaRef}
                              rows={2}
                              placeholder=""
                              onKeyDown={handleInputKeyDown}
                              className="w-full bg-transparent resize-none border-none outline-none text-xs sm:text-base text-[#0f172a] leading-relaxed relative z-10 p-0 shadow-none focus-visible:ring-0"
                            />

                            {/* Animated Blur Text Placeholder Overlay */}
                            {!inputText && (
                              <div
                                className="pointer-events-none absolute inset-0 flex items-start select-none overflow-hidden pr-2 sm:pr-4"
                              >
                                <AnimatePresence mode="wait">
                                  <motion.div
                                    key={placeholderQuestionIndex}
                                    initial={{ opacity: 0, filter: "blur(8px)", y: 4 }}
                                    animate={{ opacity: 1, filter: "blur(0px)", y: 0 }}
                                    exit={{ opacity: 0, filter: "blur(8px)", y: -4 }}
                                    transition={{ duration: 0.45, ease: "easeInOut" }}
                                    className="text-xs sm:text-sm text-slate-400 leading-normal sm:leading-relaxed flex flex-wrap items-center gap-x-1"
                                  >
                                    <ZapIcon className="w-3.5 h-3.5 text-emerald-500 inline mr-0.5 shrink-0" />
                                    {(activeQuestions[placeholderQuestionIndex % activeQuestions.length] || DEMO_QUESTIONS[0]).split(" ").map((word, wIdx) => (
                                      <motion.span
                                        key={`${placeholderQuestionIndex}-${wIdx}`}
                                        initial={{ opacity: 0, filter: "blur(6px)" }}
                                        animate={{ opacity: 1, filter: "blur(0px)" }}
                                        transition={{
                                          duration: 0.25,
                                          delay: wIdx * 0.03,
                                          ease: "easeOut",
                                        }}
                                        className="inline-block"
                                      >
                                        {word}
                                      </motion.span>
                                    ))}
                                    <span className="hidden sm:inline-flex items-center text-[10px] font-mono font-medium text-slate-500 bg-slate-100 border border-slate-200 rounded px-1.5 py-0.5 ml-1.5 self-center shadow-2xs">
                                      Tab ⇥
                                    </span>
                                  </motion.div>
                                </AnimatePresence>
                              </div>
                            )}
                          </div>

                          {/* Bottom Toolbar inside Card */}
                          <div className="flex flex-wrap items-center justify-between gap-2 pt-2.5 sm:pt-3 border-t border-slate-100/90 mt-2">
                            <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
                              {/* Language Select Option (Glassmorphism Pill) */}
                              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/40 backdrop-blur-[9px] border border-white/60 text-[11px] sm:text-xs text-slate-700 hover:bg-white/60 transition shadow-2xs">
                                <GlobeIcon className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                                <span className="text-slate-500 font-normal hidden sm:inline">
                                  {uiText.language || "Language"}:
                                </span>
                                <select
                                  value={selectedLanguageCode}
                                  onChange={(event) => {
                                    recognitionRef.current?.abort();
                                    window.speechSynthesis?.cancel();
                                    setIsListening(false);
                                    setIsSpeakingMessageId(null);
                                    setSelectedLanguageCode(event.target.value);
                                    setSelectedVoiceURI("");
                                    setVoiceMessage("");
                                    setTranslationNotice("");
                                  }}
                                  disabled={isGenerating}
                                  className="bg-transparent border-none outline-none font-medium text-slate-800 cursor-pointer text-[11px] sm:text-xs focus:ring-0 pr-1 py-0.5"
                                  aria-label="Select Language"
                                >
                                  {LANGUAGES.map((language) => (
                                    <option key={language.code} value={language.code} className="text-slate-800 bg-white">
                                      {language.code === "en" ? language.label : `${language.label} · ${language.nativeLabel}`}
                                    </option>
                                  ))}
                                </select>
                              </div>
                            </div>

                            <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
                              <button
                                type="button"
                                onClick={toggleListening}
                                disabled={!recognitionSupported || isGenerating}
                                aria-pressed={isListening}
                                className={`w-7 h-7 sm:w-8 sm:h-8 rounded-full flex items-center justify-center transition shrink-0 disabled:opacity-40 shadow-2xs ${isListening
                                  ? "bg-red-500 text-white"
                                  : "bg-emerald-600/90 hover:bg-emerald-600 text-white"
                                  }`}
                                title={isListening ? "Stop voice input" : "Start voice input"}
                              >
                                {isListening ? (
                                  <StopIcon className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                                ) : (
                                  <MicrophoneIcon className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                                )}
                              </button>

                              <ChatInputSubmit className="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-emerald-600 hover:bg-emerald-700 text-white transition disabled:bg-slate-200 disabled:text-slate-400 flex items-center justify-center p-0 shadow-sm" />
                            </div>
                          </div>
                        </ChatInput>
                      </motion.div>

                      {/* Legal Question Chips Below Card */}
                      <motion.div
                        exit={{ opacity: 0, y: 12, transition: { duration: 0.2 } }}
                        className="pt-3 sm:pt-5 px-0 space-y-3 sm:space-y-3.5"
                      >
                        {/* Most Popular Questions Pill */}
                        <div className="flex items-center justify-center">
                          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-[30px] glass-pill shadow-xs">
                            <span className="text-[11px] sm:text-xs font-semibold text-slate-700 tracking-tight">
                              {uiText.suggestedInquiries}
                            </span>
                          </div>
                        </div>

                        {/* Mobile: full-width stacked list */}
                        <div className="flex flex-col gap-2 sm:hidden">
                          {activeQuestions.slice(0, 4).map((s) => (
                            <button
                              key={s}
                              onClick={() => {
                                setInputText(s);
                                handleSendMessage(s);
                              }}
                              className="w-full text-left flex items-center gap-2.5 px-4 py-3 rounded-[24px] text-[12px] font-medium text-slate-700 glass-pill active:scale-[0.98] transition-all shadow-xs leading-snug"
                            >
                              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0 mt-px" />
                              {s}
                            </button>
                          ))}
                        </div>
                        {/* Desktop: horizontal wrap pills */}
                        <div className="hidden sm:flex flex-wrap items-center justify-center gap-2">
                          {activeQuestions.slice(0, 4).map((s) => (
                            <button
                              key={s}
                              onClick={() => {
                                setInputText(s);
                                handleSendMessage(s);
                              }}
                              className="px-4 py-1.5 rounded-[30px] text-xs font-medium text-slate-700 glass-pill transition shadow-xs max-w-[92vw] truncate"
                            >
                              {s}
                            </button>
                          ))}
                        </div>
                      </motion.div>
                    </div>

                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* ─── ACTIVE MESSAGES LIST ─────────────────────────────────────── */}
            {messages.map((msg, idx) => {
              const lastUserIndex = messages.map((m) => m.role).lastIndexOf("user");

              if (msg.role === "user") {
                const isLatestUser = idx === lastUserIndex;
                if (isTranslating) {
                  return (
                    <UserMessageSkeleton
                      key={`skeleton-user-${msg.id}`}
                      isLatestUser={isLatestUser}
                      latestTurnRef={latestTurnRef}
                    />
                  );
                }
                return (
                  <UserMessageItem
                    key={msg.id}
                    msg={msg}
                    isLatestUser={isLatestUser}
                    latestTurnRef={latestTurnRef}
                    uiText={uiText}
                    selectedLanguageCode={selectedLanguageCode}
                    translationCache={translationCacheRef}
                    setInputText={setInputText}
                    handleSendMessage={handleSendMessage}
                    isGlobalTranslating={isTranslating}
                  />
                );
              }

              if (msg.status === "loading") {
                return (
                  <div key={msg.id} className="flex items-start gap-2.5 sm:gap-3 pt-2">
                    <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-white border border-slate-200/80 p-0.5 flex items-center justify-center shrink-0 mt-0.5 shadow-2xs overflow-hidden">
                      <img src="/ipsakti_logo.png" alt="IP SAKTI Sahayak" className="w-full h-full object-contain" />
                    </div>
                    <div className="flex-1 max-w-[1080px] bg-white/95 border border-slate-200/80 rounded-2xl sm:rounded-3xl p-3.5 sm:p-5 shadow-sm">
                      <div className="flex items-center justify-between pb-3">
                        <div className="flex items-center gap-2">
                          <span className="flex items-center gap-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 dot-1" />
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 dot-2" />
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 dot-3" />
                          </span>
                          <span className="text-[11px] sm:text-xs font-medium text-slate-600">
                            {currentStepIndex >= ANALYSIS_STEPS.length
                              ? (uiText.evidenceVerified || "Evidence verified · Finalizing answer...")
                              : (uiText.analyzingEvidence || "Analyzing legal evidence...")}
                          </span>
                        </div>
                        {currentStepIndex < ANALYSIS_STEPS.length && (
                          <button
                            onClick={stopGeneration}
                            className="flex items-center gap-1 px-2 py-0.5 sm:px-2.5 sm:py-1 rounded-full border border-slate-200 text-[11px] text-slate-500 hover:bg-slate-50 transition"
                          >
                            <StopIcon className="w-3 h-3" />
                            <span>{uiText.stop || "Stop"}</span>
                          </button>
                        )}
                      </div>

                      <div className="border border-slate-100 rounded-2xl p-3 sm:p-4 bg-slate-50/50">
                        <div className="flex items-center justify-between mb-2.5">
                          <span className="text-[9px] sm:text-[10px] font-semibold text-slate-400 uppercase tracking-widest">
                            {uiText.evidenceAnalysis || "Evidence Analysis"}
                          </span>
                          <span className="text-[9px] sm:text-[10px] text-slate-400">
                            {uiText.secure || "Secure"}
                          </span>
                        </div>

                        <div className="space-y-2">
                          {ANALYSIS_STEPS.map((stepText, idx) => {
                            const isDone = idx < currentStepIndex;
                            const isCurrent = idx === currentStepIndex;
                            const localizedStep = uiText.analysisSteps?.[idx] || stepText;

                            return (
                              <div
                                key={stepText}
                                className="flex items-center gap-2 text-xs"
                              >
                                {isDone ? (
                                  <CheckIcon className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                                ) : isCurrent ? (
                                  <Loader2 className="w-3.5 h-3.5 text-emerald-600 animate-spin shrink-0" />
                                ) : (
                                  <div className="w-3.5 h-3.5 rounded-full border border-slate-200 shrink-0" />
                                )}
                                <span
                                  className={`${isDone
                                    ? "text-slate-500"
                                    : isCurrent
                                      ? "text-slate-800 font-medium"
                                      : "text-slate-300"
                                    }`}
                                >
                                  {localizedStep}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              }

              if (msg.status === "error") {
                return (
                  <div key={msg.id} className="flex items-start gap-2.5 sm:gap-3 pt-2">
                    <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-white border border-slate-200/80 p-0.5 flex items-center justify-center shrink-0 shadow-2xs overflow-hidden">
                      <img src="/ipsakti_logo.png" alt="IP SAKTI Sahayak" className="w-full h-full object-contain" />
                    </div>
                    <div
                      role="alert"
                      className="flex-1 rounded-2xl border border-red-100 bg-red-50/80 p-3.5 sm:p-4 text-xs sm:text-sm text-red-700"
                    >
                      {msg.error}
                    </div>
                  </div>
                );
              }

              if (isTranslating) {
                return <AssistantMessageSkeleton key={`skeleton-${msg.id}`} />;
              }

              const answerLanguage = languageByCode(
                msg.displayLanguageCode || "en",
              );
              return (
                <div
                  key={msg.id}
                  className="flex items-start gap-2.5 sm:gap-3 pt-2 scroll-mt-4"
                >
                  <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-white border border-slate-200/80 p-0.5 flex items-center justify-center shrink-0 mt-0.5 shadow-2xs overflow-hidden">
                    <img src="/ipsakti_logo.png" alt="IP SAKTI Sahayak" className="w-full h-full object-contain" />
                  </div>
                  <div className="flex-1 min-w-0 max-w-[1080px] bg-white/95 border border-slate-200/80 rounded-2xl sm:rounded-3xl p-3.5 sm:p-6 shadow-sm space-y-3.5 sm:space-y-4">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-1.5 text-slate-500">
                        <ShieldIcon className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                        <span className="text-[9px] sm:text-[10px] font-semibold tracking-widest uppercase">
                          {uiText.answer || "Answer"} · {answerLanguage.nativeLabel || answerLanguage.label}
                        </span>
                      </div>
                      {msg.confidence && (
                        <span className="px-2 py-0.5 rounded-full text-[9px] sm:text-[10px] font-semibold tracking-wide bg-slate-100 text-slate-600 border border-slate-200/60 uppercase shrink-0">
                          {uiText.confidenceLevels?.[msg.confidence] || msg.confidence}
                        </span>
                      )}
                    </div>

                    <AssistantMessageText
                      msg={msg}
                      selectedLanguageCode={selectedLanguageCode}
                      translationCache={translationCacheRef}
                    />

                    {msg.kind === "clarification" && msg.clarificationOptions?.length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {msg.clarificationOptions.map((option) => (
                          <span
                            key={option}
                            className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-0.5 text-[10px] sm:text-[11px] text-slate-600"
                          >
                            {option}
                          </span>
                        ))}
                      </div>
                    )}

                    <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-100 pt-2.5 sm:pt-3">
                      <div className="flex flex-wrap gap-1.5 text-[9px] sm:text-[10px] font-medium text-slate-500">
                        {msg.jurisdiction !== undefined &&
                          msg.jurisdiction !== null && (
                            <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-0.5 sm:py-1">
                              {String(msg.jurisdiction)}
                            </span>
                          )}
                        {(!msg.kind || msg.kind === "answered") && (
                          <span className="rounded-full border border-emerald-200/60 bg-emerald-50/80 px-2.5 py-0.5 sm:py-1 text-emerald-700">
                            {typeof uiText.citationsValidated === "function"
                              ? uiText.citationsValidated(msg.citations?.length || 0)
                              : `${msg.citations?.length || 0} citations validated`}
                          </span>
                        )}
                      </div>
                      <button
                        type="button"
                        onClick={() => toggleSpeech(msg)}
                        disabled={!speechSupported}
                        aria-pressed={isSpeakingMessageId === msg.id}
                        className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 px-2.5 py-1 sm:px-3 sm:py-1.5 text-[11px] sm:text-xs font-medium text-slate-600 transition hover:bg-slate-50 disabled:opacity-40"
                      >
                        {isSpeakingMessageId === msg.id ? (
                          <StopIcon className="h-3.5 w-3.5" />
                        ) : (
                          <SpeakerLoudIcon className="h-3.5 w-3.5 text-emerald-600" />
                        )}
                        {isSpeakingMessageId === msg.id
                          ? (uiText.stopAudio || "Stop audio")
                          : `${uiText.listenAnswer || "Listen"} · ${answerLanguage.nativeLabel || answerLanguage.label}`}
                      </button>
                    </div>

                    {msg.evidence && msg.evidence.length > 0 && (
                      <SourcesList
                        msg={msg}
                        uiText={uiText}
                        selectedLanguageCode={selectedLanguageCode}
                        translationCache={translationCacheRef}
                      />
                    )}

                    <p className="border-t border-slate-100 pt-2.5 sm:pt-3 text-[10px] sm:text-[11px] leading-relaxed text-slate-400">
                      {msg.caution?.detail || msg.disclaimer || uiText.legalDisclaimerText || LEGAL_DISCLAIMER}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* ─── BOTTOM FLOATING INPUT DOCK ──────────────────────────────────── */}
        <AnimatePresence>
          {(messages.length > 0 || isGenerating) && (
            <motion.div
              key="bottom-floating-dock"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              transition={{ duration: 0.25 }}
              className="relative px-3 sm:px-8 py-2.5 sm:py-3 shrink-0 z-20 bg-transparent"
            >
              <div className="max-w-[760px] mx-auto space-y-2 relative">
                <motion.div
                  layoutId="unified-chat-card"
                  transition={{ type: "spring", stiffness: 280, damping: 28 }}
                  className="w-full"
                >
                  <ChatInput
                    variant="unstyled"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onSubmit={() => handleSendMessage()}
                    loading={isGenerating}
                    onStop={stopGeneration}
                    className="glass-card-30 p-3 sm:p-4 flex flex-col items-stretch"
                  >
                    <div className="relative w-full">
                      <ChatInputTextArea
                        ref={textareaRef}
                        rows={2}
                        placeholder=""
                        onKeyDown={handleInputKeyDown}
                        className="w-full bg-transparent resize-none border-none outline-none text-xs sm:text-base text-slate-800 leading-relaxed relative z-10 p-0 shadow-none focus-visible:ring-0"
                      />

                      {!inputText && (
                        <div
                          className="pointer-events-none absolute inset-0 flex items-start select-none overflow-hidden pr-2 sm:pr-4"
                        >
                          <AnimatePresence mode="wait">
                            <motion.div
                              key={placeholderQuestionIndex}
                              initial={{ opacity: 0, filter: "blur(8px)", y: 4 }}
                              animate={{ opacity: 1, filter: "blur(0px)", y: 0 }}
                              exit={{ opacity: 0, filter: "blur(8px)", y: -4 }}
                              transition={{ duration: 0.45, ease: "easeInOut" }}
                              className="text-xs sm:text-sm text-slate-400 leading-normal sm:leading-relaxed flex flex-wrap items-center gap-x-1"
                            >
                              {(activeQuestions[placeholderQuestionIndex % activeQuestions.length] || DEMO_QUESTIONS[0]).split(" ").map((word, wIdx) => (
                                <motion.span
                                  key={`${placeholderQuestionIndex}-${wIdx}`}
                                  initial={{ opacity: 0, filter: "blur(6px)" }}
                                  animate={{ opacity: 1, filter: "blur(0px)" }}
                                  transition={{
                                    duration: 0.25,
                                    delay: wIdx * 0.03,
                                    ease: "easeOut",
                                  }}
                                  className="inline-block"
                                >
                                  {word}
                                </motion.span>
                              ))}
                              <span className="hidden sm:inline-flex items-center text-[10px] font-mono font-medium text-slate-400 bg-slate-100 border border-slate-200 rounded px-1.5 py-0.5 ml-1 self-center">
                                Tab ⇥
                              </span>
                            </motion.div>
                          </AnimatePresence>
                        </div>
                      )}
                    </div>

                    <div className="flex flex-wrap items-center justify-between w-full pt-2 sm:pt-2.5 border-t border-slate-100 mt-2 gap-1.5">
                      <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
                        {/* Language Select Option (Glassmorphism Pill) */}
                        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-white/40 backdrop-blur-[9px] border border-white/60 text-[11px] text-slate-700 hover:bg-white/60 transition shadow-2xs">
                          <GlobeIcon className="w-3 h-3 text-emerald-600 shrink-0" />
                          <select
                            value={selectedLanguageCode}
                            onChange={(event) => {
                              recognitionRef.current?.abort();
                              window.speechSynthesis?.cancel();
                              setIsListening(false);
                              setIsSpeakingMessageId(null);
                              setSelectedLanguageCode(event.target.value);
                              setSelectedVoiceURI("");
                              setVoiceMessage("");
                              setTranslationNotice("");
                            }}
                            disabled={isGenerating}
                            className="bg-transparent border-none outline-none font-medium text-slate-800 cursor-pointer text-[11px] focus:ring-0 pr-1 py-0"
                            aria-label="Select Language"
                          >
                            {LANGUAGES.map((language) => (
                              <option key={language.code} value={language.code} className="text-slate-800 bg-white">
                                {language.label}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>

                      <div className="flex items-center gap-1.5 sm:gap-2">
                        <button
                          type="button"
                          onClick={toggleListening}
                          disabled={!recognitionSupported || isGenerating}
                          aria-pressed={isListening}
                          className={`w-7 h-7 sm:w-8 sm:h-8 rounded-full flex items-center justify-center transition shrink-0 disabled:opacity-40 ${isListening
                            ? "bg-red-500 text-white"
                            : "text-slate-500 hover:text-slate-800 hover:bg-slate-100 border border-transparent"
                            }`}
                          title={isListening ? "Stop voice input" : "Start voice input"}
                        >
                          {isListening ? (
                            <StopIcon className="w-3.5 h-3.5" />
                          ) : (
                            <MicrophoneIcon className="w-3.5 h-3.5" />
                          )}
                        </button>

                        <ChatInputSubmit className="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-emerald-600 text-white hover:bg-emerald-700 transition disabled:bg-slate-100 disabled:text-slate-300 flex items-center justify-center p-0 shadow-2xs" />
                      </div>
                    </div>
                  </ChatInput>
                </motion.div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ─── INTERACTIVE STEPPER POPUP MODAL ─────────────────────────────── */}
        <AnimatePresence>
          {showGuide && (
            <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 overflow-y-auto">
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setShowGuide(false)}
                className="fixed inset-0 bg-slate-900/10 backdrop-blur-[2px] transition-opacity"
              />

              <motion.div
                initial={{ opacity: 0, scale: 0.96, y: 12 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.96, y: 12 }}
                transition={{ duration: 0.25, ease: "easeOut" }}
                className="relative w-full max-w-2xl bg-white/95 backdrop-blur-xl rounded-2xl sm:rounded-3xl shadow-2xl shadow-emerald-950/10 border border-slate-200/90 z-10 overflow-hidden my-auto"
              >
                <div className="flex items-center justify-between px-4 sm:px-6 py-3.5 sm:py-4 border-b border-slate-100 bg-white">
                  <div className="flex items-center gap-2.5">
                    <div>
                      <h3 className="text-xs sm:text-sm font-semibold text-slate-900 leading-none">
                        {uiText.tutorialTitle}
                      </h3>
                      <p className="text-[10px] sm:text-[11px] text-slate-400 mt-0.5">
                        {uiText.tutorialSubtitle}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setShowGuide(false)}
                      className="w-7 h-7 rounded-full flex items-center justify-center text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition"
                      title="Close"
                    >
                      <Cross2Icon className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>

                <div className="p-2 sm:p-4">
                  <Stepper
                    initialStep={1}
                    showSkipButton={true}
                    skipButtonText={uiText.skipIntro}
                    onSkip={() => {
                      setShowGuide(false);
                      setGuideCompleted(true);
                    }}
                    onFinalStepCompleted={() => {
                      setShowGuide(false);
                      setGuideCompleted(true);
                    }}
                    backButtonText={uiText.previous}
                    nextButtonText={uiText.continue}
                    stepCircleContainerClassName="!border-none !shadow-none !max-w-none"
                  >
                    <Step>
                      <div className="space-y-3">
                        <h2 className="text-base sm:text-lg font-semibold text-slate-900 tracking-tight">
                          Search Indian Statutory Law
                        </h2>
                        <p className="text-xs text-slate-500 leading-relaxed">
                          {uiText.statutoryIntelligenceDesc || "Statutory intelligence and citation-grounded guidance across Indian laws, acts, and regulations."}
                        </p>
                        <div className="pt-1 text-xs">
                          <div className="p-3.5 rounded-2xl border border-slate-200/80 bg-slate-50/60 w-full space-y-2">
                            <div className="flex items-center gap-1.5">
                              <BookmarkIcon className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                              <span className="font-semibold text-slate-800 text-xs">Multiple Official Acts &amp; Regulations</span>
                            </div>
                            <div className="flex flex-wrap gap-1.5 pt-0.5">
                              {[
                                "Patents Act",
                                "Trade Marks Act",
                                "Copyright Act",
                                "Biological Diversity Act",
                                "Designs Act",
                                "IP Rights",
                                "Drugs & Cosmetics",
                                "FSSAI Regulations",
                                "& More",
                              ].map((act) => (
                                <span
                                  key={act}
                                  className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200/60"
                                >
                                  {act}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    </Step>

                    <Step>
                      <div className="space-y-3">
                        <h2 className="text-base sm:text-lg font-semibold text-slate-900 tracking-tight">
                          Inspect Citations &amp; Try A Query
                        </h2>
                        <p className="text-xs text-slate-500 leading-relaxed">
                          Every answer links directly to official rules and gazette text. Pick a query to start:
                        </p>
                        <div className="space-y-1.5 pt-0.5">
                          {activeQuestions.slice(0, 3).map((q) => (
                            <button
                              key={q}
                              type="button"
                              onClick={() => {
                                setInputText(q);
                                setShowGuide(false);
                                setGuideCompleted(true);
                                handleSendMessage(q);
                              }}
                              className="w-full text-left p-2.5 rounded-xl border border-slate-200 bg-white hover:bg-emerald-50/50 hover:border-emerald-200 transition text-xs text-slate-800 flex items-center justify-between group"
                            >
                              <span className="font-medium truncate pr-2 text-[11px]">{q}</span>
                              <span className="text-[10px] text-emerald-600 font-semibold group-hover:translate-x-0.5 transition-all shrink-0">
                                Ask →
                              </span>
                            </button>
                          ))}
                        </div>
                      </div>
                    </Step>
                  </Stepper>
                </div>
              </motion.div>
            </div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
