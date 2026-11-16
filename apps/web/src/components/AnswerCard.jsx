"use client";

import React from "react";
import CautionBanner from "./CautionBanner";

export default function AnswerCard({ result, onSelectCitation }) {
  if (!result) return null;

  const isAbstained = result.status === "abstained";
  const answer = result.answer || {};
  const text = answer.text || "";
  const confidence = answer.confidence;
  const cautions = result.cautions || [];

  // Parse text to render [n] citation markers as clickable badges
  const renderFormattedAnswer = (rawText) => {
    if (!rawText) return null;
    const parts = rawText.split(/(\[\d+\])/g);
    return parts.map((part, index) => {
      const match = part.match(/\[(\d+)\]/);
      if (match) {
        const markerNum = parseInt(match[1], 10);
        return (
          <button
            key={index}
            type="button"
            onClick={() => onSelectCitation && onSelectCitation(markerNum)}
            className="inline-flex items-center justify-center font-mono font-bold text-xs bg-citation text-citation-text hover:bg-citation/80 border border-citation-text/30 px-1.5 py-0.5 mx-1 rounded-sm transition-colors cursor-pointer"
            title={`Jump to Source [${markerNum}]`}
          >
            {part}
          </button>
        );
      }
      return <span key={index}>{part}</span>;
    });
  };

  return (
    <section className="wireframe-box p-6 border-b border-wireframe space-y-4 max-w-[1040px] mx-auto w-full">
      <div className="flex items-center justify-between border-b border-wireframe/60 pb-2">
        <h2 className="text-sm font-mono uppercase tracking-wider text-prose font-semibold">
          ANSWER
        </h2>
        <span
          className={`text-[11px] font-mono px-2 py-0.5 border uppercase tracking-wider ${isAbstained
            ? "bg-amber-950/40 text-amber-400 border-amber-800/60"
            : "bg-emerald-950/40 text-emerald-400 border-emerald-800/60"
            }`}
        >
          {isAbstained ? "ABSTAINED" : "ANSWERED"}
        </span>
      </div>

      {/* Answer Body */}
      <div className="text-sm sm:text-base leading-relaxed font-sans text-prose/95">
        {renderFormattedAnswer(text)}
      </div>

      {/* Confidence Indicator */}
      {!isAbstained && confidence && (
        <div className="pt-2 flex items-center gap-2 font-mono text-xs text-subtext">
          <span className="uppercase text-muted">Confidence:</span>
          <span
            className={`font-semibold uppercase ${confidence === "HIGH"
              ? "text-emerald-400"
              : confidence === "LOW"
                ? "text-amber-400"
                : "text-blue-300"
              }`}
          >
            {confidence.charAt(0) + confidence.slice(1).toLowerCase()}
          </span>
          <span className="text-zinc-600 text-[10px]">
            (Derived from provenance & statutory authority)
          </span>
        </div>
      )}

      {/* Cautions (if any) */}
      <CautionBanner cautions={cautions} />
    </section>
  );
}
