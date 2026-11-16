"use client";

import React from "react";

export default function HeaderCard() {
  return (
    <header className="wireframe-box p-6 rounded-none text-center border-b-2 border-wireframe relative overflow-hidden">
      <div className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-slate-500/30 to-transparent" />
      <h1 className="text-2xl sm:text-3xl font-bold tracking-wider text-prose uppercase font-mono">
        IP-SAKTI
      </h1>
      <p className="text-subtext text-xs sm:text-sm mt-1.5 tracking-wide font-mono uppercase">
        Grounded Intellectual Property AI
      </p>
    </header>
  );
}
