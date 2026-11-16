"use client";

import React from "react";
import { Shield } from "lucide-react";

export default function Disclaimer() {
  return (
    <footer className="mt-8 mb-12 text-center text-xs font-mono text-muted space-y-1">
      <div className="flex items-center justify-center gap-1.5 text-zinc-500">
        <Shield className="w-3.5 h-3.5" />
        <span className="uppercase tracking-wider font-medium">System Disclaimer</span>
      </div>
      <p className="max-w-2xl mx-auto text-zinc-500 leading-normal px-4">
        IP-SAKTI is an information and research tool that answers strictly from verified statutory texts.
        It is not legal advice and does not substitute for a qualified legal practitioner.
      </p>
    </footer>
  );
}
