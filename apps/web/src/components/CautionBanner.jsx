"use client";

import React from "react";
import { AlertTriangle } from "lucide-react";

export default function CautionBanner({ cautions = [] }) {
  if (!cautions || cautions.length === 0) return null;

  return (
    <div className="border border-ambernotice-border/60 bg-ambernotice/40 p-4 rounded-none my-4 space-y-2">
      <div className="flex items-center gap-2 text-amber-400 font-mono text-xs uppercase tracking-wider font-semibold">
        <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
        <span>Source Caution Notice</span>
      </div>
      <ul className="space-y-1.5 pl-6 list-disc text-amber-200/90 text-xs font-sans leading-relaxed">
        {cautions.map((caution, idx) => (
          <li key={idx}>{caution}</li>
        ))}
      </ul>
    </div>
  );
}
