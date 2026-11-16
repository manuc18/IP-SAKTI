"use client";
import React from "react";
import {
  MessageSquare,
  BookOpen,
  Plus,
  Trash2,
  X,
  PanelLeft,
} from "lucide-react";

export function Sidebar({
  isOpen,
  onClose,
  uiText,
  activeItem = "chat",
  onSelectItem,
  onOpenTutorial,
  chatSessions = [],
  activeSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  onClearConversation,
  isTranslating = false,
}) {
  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          onClick={onClose}
          className="fixed inset-0 bg-slate-900/30 backdrop-blur-xs z-40 lg:hidden transition-opacity"
        />
      )}

      {/* Sidebar container - Light Mode Glassmorphism with smooth collapse animation */}
      <aside
        className={`fixed lg:static top-0 left-0 h-full z-50 flex flex-col glass-sidebar text-slate-800 transition-all duration-300 ease-in-out shadow-sm lg:shadow-none overflow-hidden ${
          isOpen
            ? "w-[280px] sm:w-[290px] translate-x-0 opacity-100"
            : "w-0 -translate-x-full lg:translate-x-0 lg:w-0 opacity-0 pointer-events-none border-none"
        }`}
      >
        <div className="w-[280px] sm:w-[290px] h-full flex flex-col shrink-0">
          {/* Top Header branding with original logo & PanelLeft toggle */}
          <div className="p-4 sm:p-5 flex items-center justify-between border-b border-white/40 bg-white/30 backdrop-blur-[9px]">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-9 h-9 rounded-full bg-white/90 border border-white/60 p-0.5 flex items-center justify-center shrink-0 shadow-2xs overflow-hidden">
                <img
                  src="/ipsakti_logo.png"
                  alt="IP-SAKTI Sahayak Logo"
                  className="w-full h-full object-contain rounded-full"
                />
              </div>
              <div className="min-w-0">
                <h1 className="text-[14.5px] font-bold text-slate-900 tracking-tight truncate leading-snug">
                  {uiText.appTitle}
                </h1>
                <p className="text-[11px] font-medium text-emerald-700 truncate">
                  {uiText.appSubtitle}
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-xl text-slate-500 hover:text-slate-900 hover:bg-white/60 transition-all duration-200 flex items-center justify-center shadow-2xs group active:scale-95 cursor-pointer"
              title="Close sidebar"
              aria-label="Close sidebar"
            >
              <PanelLeft className="w-5 h-5 text-slate-600 group-hover:text-slate-900 transition-transform duration-200 group-hover:scale-105" />
            </button>
          </div>

        {/* Action Button: New Chat */}
        <div className="p-3 border-b border-white/30 bg-white/20 backdrop-blur-[9px]">
          <button
            type="button"
            onClick={() => {
              onNewChat?.();
              if (typeof window !== "undefined" && window.innerWidth < 1024) {
                onClose?.();
              }
            }}
            className="w-full flex items-center justify-center gap-2 px-3.5 py-2.5 rounded-[20px] bg-emerald-600/90 hover:bg-emerald-600 text-white font-semibold text-xs shadow-xs border border-white/40 transition hover:scale-[1.01] active:scale-[0.98]"
            title={uiText.newChat}
          >
            <Plus className="w-4 h-4" />
            <span>{uiText.newChat}</span>
          </button>
        </div>

        {/* Scrollable Body: Chat, Tutorial & Recent Chats */}
        <div className="flex-1 overflow-y-auto overflow-x-hidden p-3 space-y-3.5 custom-scrollbar">
          {/* Main Primary Links: Chat & Tutorial */}
          <nav className="space-y-1">
            <button
              onClick={() => {
                onSelectItem?.("chat");
                if (typeof window !== "undefined" && window.innerWidth < 1024) {
                  onClose?.();
                }
              }}
              className={`w-full flex items-center gap-2.5 px-3.5 py-2 rounded-[20px] text-xs font-semibold transition text-left backdrop-blur-[9px] ${
                activeItem === "chat"
                  ? "bg-white/70 text-emerald-950 border border-white/80 shadow-xs"
                  : "text-slate-600 hover:text-slate-900 hover:bg-white/40"
              }`}
            >
              <MessageSquare className="w-4 h-4 text-emerald-600 shrink-0" />
              <span className="truncate">Chat</span>
            </button>

            <button
              onClick={() => {
                onOpenTutorial?.();
                if (typeof window !== "undefined" && window.innerWidth < 1024) {
                  onClose?.();
                }
              }}
              className="w-full flex items-center gap-2.5 px-3.5 py-2 rounded-[20px] text-xs font-medium text-slate-600 hover:text-slate-900 hover:bg-white/40 transition text-left"
            >
              <BookOpen className="w-4 h-4 text-slate-500 shrink-0" />
              <span className="truncate">{uiText.tutorial || "Tutorial"}</span>
            </button>
          </nav>

          {/* Recent Chats Section */}
          <div className="pt-1 space-y-2">
            <div className="flex items-center justify-between px-1.5">
              <span className="text-[11px] font-bold tracking-wider text-slate-500 uppercase flex items-center gap-1.5">
                <MessageSquare className="w-3.5 h-3.5 text-slate-400" />
                {uiText.recentChats}
              </span>
            </div>

            {/* List of recent chat sessions */}
            <div className="space-y-1 overflow-y-auto pr-0.5">
              {isTranslating ? (
                <div className="space-y-1.5 pr-0.5">
                  {(chatSessions.length > 0 ? chatSessions : [1, 2]).map((_, idx) => (
                    <div
                      key={`skeleton-session-${idx}`}
                      className="w-full flex items-center gap-2 px-3 py-2 rounded-[18px] bg-white/40 border border-white/50 animate-pulse"
                    >
                      <div className="w-3.5 h-3.5 rounded bg-slate-200/80 shrink-0" />
                      <div className="h-3 bg-slate-200/80 rounded-full w-3/4" />
                    </div>
                  ))}
                </div>
              ) : chatSessions.length === 0 ? (
                <div className="px-3 py-4 text-center text-xs text-slate-400 italic bg-white/30 backdrop-blur-[9px] rounded-[20px] border border-dashed border-white/50">
                  {uiText.emptyRecentChats}
                </div>
              ) : (
                chatSessions.map((session) => (
                  <div
                    key={session.id}
                    onClick={() => {
                      onSelectSession?.(session.id);
                      if (typeof window !== "undefined" && window.innerWidth < 1024) {
                        onClose?.();
                      }
                    }}
                    className={`group w-full flex items-center justify-between px-3 py-2 rounded-[18px] text-xs transition cursor-pointer text-left backdrop-blur-[9px] ${
                      activeSessionId === session.id
                        ? "bg-white/70 text-emerald-950 border border-white/80 font-semibold shadow-xs"
                        : "text-slate-600 hover:text-slate-900 hover:bg-white/50 border border-transparent hover:border-white/50"
                    }`}
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <MessageSquare
                        className={`w-3.5 h-3.5 shrink-0 ${
                          activeSessionId === session.id
                            ? "text-emerald-600"
                            : "text-slate-400"
                        }`}
                      />
                      <span className="truncate">{session.title || "Legal Consultation"}</span>
                    </div>
                    {onDeleteSession && (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteSession(session.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 p-1 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded transition shrink-0"
                        title="Delete chat"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Bottom Section: Clear Conversation with dustbin icon */}
        <div className="p-3 border-t border-white/40 bg-white/30 backdrop-blur-[9px]">
          <button
            type="button"
            onClick={() => {
              if (onClearConversation) {
                onClearConversation();
              } else if (activeSessionId && onDeleteSession) {
                onDeleteSession(activeSessionId);
              } else if (onNewChat) {
                onNewChat();
              }
            }}
            className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-xl text-xs font-medium text-slate-600 hover:text-red-600 hover:bg-red-50/80 transition group cursor-pointer"
            title={uiText?.clearConversation || "Clear conversation"}
          >
            <div className="w-6 h-6 rounded-lg bg-slate-100 group-hover:bg-red-100 flex items-center justify-center text-slate-400 group-hover:text-red-500 transition shrink-0">
              <Trash2 className="w-3.5 h-3.5" />
            </div>
            <span className="truncate">{uiText?.clearConversation || "Clear conversation"}</span>
          </button>
        </div>
      </div>
    </aside>
    </>
  );
}
