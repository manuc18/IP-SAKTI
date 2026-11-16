"use client";

import React from "react";
import { Loader2 } from "lucide-react";

export default function QueryCard({ question, setQuestion, onSubmit, loading, sampleQuestions }) {
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSubmit();
    }
  };

  return (
    <section className="wireframe-box p-6 border-b border-wireframe space-y-4">
      <div className="flex items-center justify-between">
        <label
          htmlFor="legal-query-input"
          className="block text-sm font-mono uppercase tracking-wider text-prose font-medium"
        >
          Ask your IP legal question
        </label>
        <span className="text-[11px] font-mono text-muted uppercase tracking-tight">
          Strict Statutory Evidence
        </span>
      </div>

      <div className="relative">
        <textarea
          id="legal-query-input"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your legal query (e.g., How is an application for trademark registration filed?)..."
          rows={2}
          disabled={loading}
          className="w-full bg-[#0a0c10] border border-wireframe rounded-none px-4 py-3 text-prose placeholder:text-muted focus:outline-none focus:border-wireframe-bright focus:ring-1 focus:ring-wireframe-bright font-sans text-sm resize-none transition-all disabled:opacity-50 shadow-inner"
        />
      </div>

      {/* Quick Test Chips */}
      {sampleQuestions && sampleQuestions.length > 0 && (
        <div className="pt-1">
          <div className="text-[11px] font-mono text-muted uppercase tracking-wider mb-2">
            Suggested Verification Tests:
          </div>
          <div className="flex flex-wrap gap-2">
            {sampleQuestions.map((sample, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => setQuestion(sample.query)}
                disabled={loading}
                className="text-xs font-mono px-2.5 py-1.5 bg-[#171a22] hover:bg-[#1f2430] text-subtext hover:text-prose border border-wireframe/80 rounded-none transition-all text-left truncate max-w-full"
                title={sample.query}
              >
                <span className="text-zinc-500 mr-1.5 font-bold">[{sample.label}]</span>
                {sample.query}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Submit Button */}
      <div className="flex justify-center pt-2">
        <button
          type="button"
          onClick={onSubmit}
          disabled={loading || !question.trim()}
          className="font-mono text-sm uppercase tracking-wider px-8 py-2.5 bg-[#171b24] hover:bg-[#202532] text-prose border border-wireframe hover:border-wireframe-bright transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2 group shadow-sm active:translate-y-[1px]"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin text-zinc-400" />
              <span>Analyzing Evidence...</span>
            </>
          ) : (
            <span>[ Ask IP-SAKTI ]</span>
          )}
        </button>
      </div>
    </section>
  );
}
