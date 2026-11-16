"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronUp, FileText, CheckCircle } from "lucide-react";

export default function SourcesCard({ citations = [], activeCitationId }) {
  const [expandedIds, setExpandedIds] = useState({ 1: true }); // Default first open

  if (!citations || citations.length === 0) return null;

  const toggleExpand = (id) => {
    setExpandedIds((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  return (
    <section className="wireframe-box p-6 space-y-4">
      <div className="flex items-center justify-between border-b border-wireframe/60 pb-2">
        <h2 className="text-sm font-mono uppercase tracking-wider text-prose font-semibold">
          SOURCES
        </h2>
        <span className="text-[11px] font-mono text-muted uppercase">
          {citations.length} Grounded {citations.length === 1 ? "Provision" : "Provisions"}
        </span>
      </div>

      <div className="space-y-4 pt-1">
        {citations.map((cite) => {
          const isExpanded = !!expandedIds[cite.id];
          const isHighlighted = activeCitationId === cite.id;

          return (
            <div
              key={cite.id}
              id={`source-${cite.id}`}
              className={`border transition-all p-4 rounded-none ${
                isHighlighted
                  ? "border-citation-text bg-[#161a25]"
                  : "border-wireframe/80 bg-[#0e1016]"
              }`}
            >
              {/* Citation Title & Locator */}
              <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2">
                <div className="flex items-baseline gap-2.5">
                  <span className="font-mono font-bold text-xs bg-citation text-citation-text border border-citation-text/30 px-2 py-0.5 rounded-sm">
                    {cite.marker || `[${cite.id}]`}
                  </span>
                  <span className="font-sans font-semibold text-sm sm:text-base text-prose">
                    {cite.document}
                  </span>
                </div>

                {/* Metadata badges */}
                <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono text-subtext">
                  <span className="text-zinc-300 font-medium">{cite.locator}</span>
                  {cite.page && <span>· p. {cite.page}</span>}
                  <span
                    className={`px-1.5 py-0.2 border text-[10px] uppercase ${
                      cite.source_status === "CURRENT"
                        ? "text-emerald-400 border-emerald-800/60 bg-emerald-950/20"
                        : "text-amber-400 border-amber-800/60 bg-amber-950/20"
                    }`}
                  >
                    {cite.source_status}
                  </span>
                  <span className="text-zinc-500">Level {cite.provenance_level}</span>
                </div>
              </div>

              {/* View Evidence Toggle */}
              <div className="mt-3">
                <button
                  type="button"
                  onClick={() => toggleExpand(cite.id)}
                  className="inline-flex items-center gap-1.5 text-xs font-mono text-zinc-400 hover:text-prose transition-colors cursor-pointer py-1"
                >
                  <span>{isExpanded ? "Hide Evidence" : "View Evidence"}</span>
                  {isExpanded ? (
                    <ChevronUp className="w-3.5 h-3.5 text-zinc-400" />
                  ) : (
                    <ChevronDown className="w-3.5 h-3.5 text-zinc-400" />
                  )}
                </button>
              </div>

              {/* Evidence Box */}
              {isExpanded && cite.evidence_text && (
                <div className="mt-2.5 pt-2.5 border-t border-wireframe/60">
                  <div className="bg-[#090b0e] border border-wireframe/80 p-3.5 rounded-none font-mono text-xs text-zinc-300 leading-relaxed overflow-x-auto whitespace-pre-wrap selection:bg-zinc-800">
                    <div className="text-[10px] uppercase text-zinc-500 tracking-wider mb-1 font-semibold flex items-center gap-1.5">
                      <FileText className="w-3 h-3 text-zinc-500" />
                      <span>Verbatim Corpus Extract</span>
                    </div>
                    {cite.evidence_text}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
