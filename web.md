# IP-SAKTI — Modern Centered AI Assistant Web Interface Reference (`web.md`)

> **Comprehensive UI & Architecture Reference Document**  
> Built for **IP-SAKTI — Citation-Grounded Legal AI** over Indian Intellectual Property & Biodiversity Law.  
> Directly reflects the single-column centered conversational interface (no sidebar), exact extracted color tokens, evidence-first analysis pipeline, and verified statutory provenance inspection.

---

## 1. Executive Summary & Design Vision

IP-SAKTI is an evidence-first, citation-grounded Intellectual Property legal assistant for Indian statutory law. Unlike generic LLM chat interfaces that hallucinate legal advice, IP-SAKTI adheres strictly to **"Grounded or Silent"**: every claim sentence must tie directly to verified statutory evidence, or the system safely abstains.

The web application reflects this philosophy visually through:
* **Clean Single-Column Workspace (No Sidebar)**: A focused, distraction-free conversational canvas where queries and answers are centered.
* **Evidence-First Analysis Step Loader**: During generation, the UI transparently surfaces the internal RAG stages (`Query processing` → `Evidence retrieval` → `Reranking sources` → `Selecting evidence` → `Generating grounded answer` → `Validating citations`) so users see the legal validation pipeline working securely in real time.
* **First-Class Provenance & Verification**: Citations (`[1]`, `[2]`) are interactive links directly tied to underlying statutory records. Each source displays publisher provenance levels (`A`–`D`), validation status (`VERIFIED`), and an expandable drawer to view verbatim statutory text and physical page numbers.
* **Pristine Light Theme**: Pure white background (`#ffffff`), subtle borders (`#e2e8f0`), crisp Inter typography, and charcoal accents (`#0f172a`).

---

## 2. Visual Layout & UI Structure

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ [🧁 GPT-5 ▾]                                                        [+ New Chat] [•••] [SG] │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│                                                        ┌─────────────────────────────┐ │
│                                                        │ hi                          │ │
│                                                        └─────────────────────────────┘ │
│                                                             ✎ Edit  ↻ Resend     [JD]  │
│                                                                                        │
│  ┌────┐ ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ AI │ │ ••• IP-SAKTI is analyzing legal evidence...                      [⏸ Pause] │  │
│  └────┘ │ ┌───────────────────────────────────────────────────────────────────────┐ │  │
│         │ │ EVIDENCE-FIRST ANALYSIS                               Working securely│ │  │
│         │ │ ✓ Query processing                                                    │ │  │
│         │ │ ✓ Evidence retrieval                                                  │ │  │
│         │ │ ✓ Reranking sources                                                   │ │  │
│         │ │ ✓ Selecting evidence                                                  │ │  │
│         │ │ ✓ Generating grounded answer                                          │ │  │
│         │ │ ◌ Validating citations                                                │ │  │
│         │ └───────────────────────────────────────────────────────────────────────┘ │  │
│         └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                        │
│  ┌────┐ ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ AI │ │ 🛡️ ANSWER                                                CONFIDENCE: HIGH │  │
│  └────┘ │ Based on the available test evidence, the request is eligible for review  │  │
│         │ pursuant to official rules [1].                                           │  │
│         │                                                                           │  │
│         │ ⚠️ Source notice                                                           │  │
│         │ Example mode: this result uses sample evidence only.                      │  │
│         │                                                                           │  │
│         │ 📄 SOURCES                                                                │  │
│         │ ┌───────────────────────────────────────────────────────────────────────┐ │  │
│         │ │ [1] Sample Legal Evidence                            View evidence ▾  │ │  │
│         │ │     Example record · Section 1                                        │ │  │
│         │ │     [VERIFIED]  [PROVENANCE A]                                        │ │  │
│         │ └───────────────────────────────────────────────────────────────────────┘ │  │
│         └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│             ┌────────────────────────────────────────────────────────────┐             │
│             │ How can I help you today?                                  │             │
│             │                                                            │             │
│             │ [+]                                                [🎤] [➔]│             │
│             └────────────────────────────────────────────────────────────┘             │
│                       AI can make mistakes. Check important info.                      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Extracted Design Tokens & Styling

| Token / Element | Exact Hex / Class | Purpose |
|---|---|---|
| **Canvas Background** | `#ffffff` | Pure white backdrop for the whole application |
| **Top Navigation** | `h-14 bg-white` | Clean top bar with model dropdown, `+ New Chat`, `•••`, and `SG` |
| **Primary Charcoal** | `#0f172a` | Charcoal accents: user profile badge, AI avatar, active send button, user bubble |
| **Text Primary** | `#0f172a` | Section titles, user messages, answer headings |
| **Text Secondary** | `#334155` | Source markers, status text |
| **Text Subtitle** | `#64748b` | Locators, icon colors, analysis step labels |
| **Text Muted** | `#94a3b8` | Placeholders, timestamps, disclaimer |
| **Borders** | `#e2e8f0` | Subtle 1px borders on pills, input cards, and evidence boxes |
| **Input Card** | `rounded-[24px] border-[#e2e8f0] shadow-[0_2px_8px_rgba(0,0,0,0.03)]` | Centered floating prompt container |
| **Notice Banner** | `#fffbeb` (border `#fef3c7`, text `#92400e`) | Amber banner for caveats or staleness notifications |
| **Verified Badge** | `#ecfdf5` (border `#a7f3d0`, text `#047857`) | Mint/emerald badge for verified statutory citations |
| **Provenance Badge** | `#f1f5f9` (text `#475569`) | Soft gray badge for evidence provenance tier |

---

## 4. End-to-End System Architecture

```text
┌────────────────────────────────────────────────────────┐
│             Next.js Web Frontend (apps/web)            │
│  Port 3001 · Single-Column Centered · No Sidebar       │
└──────────────────────────┬─────────────────────────────┘
                           │ POST /api/ask { question }
                           ▼
┌────────────────────────────────────────────────────────┐
│           Express API Gateway (apps/server)            │
│  Port 5001 · Request Validation · CORS · Resilience    │
└──────────────────────────┬─────────────────────────────┘
                           │ POST /api/ask { question }
                           ▼
┌────────────────────────────────────────────────────────┐
│             Python Core Engine (app.server)            │
│  Port 8000 · 3,073 Chunks · BGE-M3 Dense + BM25 Boost │
│  Feature Reranker · Groq LLM · Grounding Gate          │
└──────────────────────────┬─────────────────────────────┘
                           │ Grounded Answer + Citations
                           ▼
┌────────────────────────────────────────────────────────┐
│             Frontend Visual State Render               │
│  Answer Card · Confidence Badge · Expandable Sources   │
└────────────────────────────────────────────────────────┘
```

---

## 5. Chat Lifecycle & Clean State

* **Initial State**: Always starts fresh (`0 messages`) displaying the `New Chat` hero, metadata (`Updated 1 second ago · 0 messages`), quick suggestion tags (`Certified`, `Personalized`, `Experienced`, `Helpful`), and dashed container (`"No messages yet. Say hello to start."`).
* **No Old Mock History**: All previous placeholder messages and legacy sidebar chat records are removed.
* **Instant Reset**:
  * Clicking `+ New Chat` in the top bar instantly clears any active messages and resets the view to the empty state.
  * Clicking the `•••` action menu provides a `Clear conversation` action to purge current chat messages at any time.

### Evidence-First Analysis Synchronization (Single-Cycle)
* **No Endless Loop**: The 6 RAG pipeline stages (`Query processing` → `Evidence retrieval` → `Reranking sources` → `Selecting evidence` → `Generating grounded answer` → `Validating citations`) advance sequentially through timed realistic phases.
* **Single Spin Per Step**: Each step spins exactly once during its active window. Once completed, it locks to a green checkmark (`✓`).
* **Clean Completion**: The final step (`Validating citations`) holds active until the backend response resolves. When data arrives, all 6 steps are checked (`✓`), followed by a smooth transition directly into the completed statutory Answer Card.

### View Anchoring & Responsive Visibility (Laptop & Mobile)
* **Turn-Anchored Viewport**: When a user submits a question or an AI answer finishes, the viewport automatically scrolls and anchors to the **top of the active turn** (`block: "start"` with `scroll-mt-4`), ensuring the user's prompt and the AI Answer heading/prose remain **prominently visible on screen**.
* **Remaining Content Flows Below**: Extensive source drawers, verified badges, and statutory excerpts flow naturally underneath, accessible via smooth scrolling without obscuring or pushing the primary answer text off screen.
* **Full Mobile & Laptop Responsiveness**:
  * Fluid padding (`px-3 sm:px-8 md:px-12`) prevents mobile clipping.
  * Flex children enforce `min-w-0` and `break-words` to eliminate horizontal overflow.
  * Bottom floating input dock adapts seamlessly across narrow mobile viewports and large desktop screens.

### Independent Source Evidence State
* **Isolated `SourceItem` Component**: Each statutory evidence source card encapsulates its own independent boolean expansion state (`useState(false)`).
* **Zero Cross-Card or Cross-Turn Interference**: Expanding evidence on Source [1] toggles strictly that single card. Subsequent sources in the same message or in future chat turns will never inherit or mirror the unhidden state.
* **Controlled Scroll Containers**: Expanded excerpts are constrained within `max-h-60 overflow-y-auto`, ensuring voluminous schedules or multi-section provisions do not displace the rest of the interface.

### Interactive Onboarding Stepper Modal (React Bits Integration with Skip Option)
* **Modal Popup Dialog**:
  * Rendered inside an animated `<AnimatePresence>` overlay with a soft backdrop (`bg-slate-900/40 backdrop-blur-xs`).
  * Features spring scale/fade animations (`opacity: 0 -> 1`, `scale: 0.96 -> 1`, `y: 12 -> 0`).
  * Can be dismissed via outside click, top-right `X` icon button, header `Skip` button, or the Stepper footer's `Skip intro` button.
* **Skip Capabilities**:
  * `onSkip` prop and `showSkipButton={true}` integrated into `<Stepper />`.
  * Users can immediately skip the introduction at any step and jump straight into the chat workspace.
* **Step-by-Step Educational Narrative**:
  * **Step 1 (Statutory Grounding)**: Highlights the specialized statutory corpus (Patents Act 1970, Trade Marks Rules 2017, Copyright Act 1957, Biodiversity Act 2002) and the strict "Grounded or Silent" guarantee.
  * **Step 2 (Real-Time Pipeline)**: Explains the multi-stage pipeline: hybrid dense BGE-M3 + lexical BM25 retrieval across 3,073 chunks, feature reranking, and citation validation.
  * **Step 3 (Inspect Citations)**: Demonstrates interactive citations `[1]`, `[2]`, mint `VERIFIED` badges, and expandable source drawers revealing verbatim statutory excerpts.
  * **Step 4 (Try A Statutory Question)**: Provides one-click example legal queries from the rehearsed demo suite that close the guide and execute immediately.
* **Navigation & Accessibility**: Features a persistent `Guide` button in the top navigation bar to reopen the popup modal at any time.


### Animated Blur Text Placeholder (Rotating Legal Queries)
* **Rehearsed Statutory Questions**:
  1. *"How must an application for registration of a trade mark be signed?"*
  2. *"Is the mere discovery of a new form of a known substance patentable in India?"*
  3. *"What is the duration of the registration of a trade mark and when must it be renewed?"*
  4. *"What is the term of copyright in a literary work published in the lifetime of the author?"*
  5. *"Who may apply for the registration of a geographical indication?"*
* **Word-by-Word Blur Animation**:
  * Driven by `motion/react` and `AnimatePresence (mode="wait")`.
  * Successive questions rotate smoothly every 4.5 seconds when the input is idle.
  * Each question arrives with a word-by-word staggered blur-in transition (`filter: blur(6px) -> blur(0px)`, `opacity: 0 -> 1`), creating an organic, premium typing feel.
  * Clicking the animated placeholder focuses the underlying native textarea instantly, allowing seamless user typing without lag.

### Perplexity-Style Centered Hero Input Dock & Position Transition
* **Centered Hero Mode (Zero Messages State)**:
  * In the empty/landing state (`messages.length === 0`), the search dock is centered prominently in the hero section (`min-h-[70vh] flex flex-col items-center justify-center`).
  * Displays the clean tagline: *"What do you want to know?"* and *"Indian statutory intelligence across Patents, Trade Marks, Copyright & Biodiversity law."*
  * The input card features an integrated bottom toolbar with a file attach (`+`) button, statutory badge (`⚖️ Indian Statutory Law`), voice input (`Mic`), and send arrow (`Navigation`).
  * Quick-starter suggestion chips (`Certified`, `Personalized`, `Experienced`, `Helpful`) and the `Onboarding Guide` trigger are neatly centered beneath the input card.
* **Transition to Bottom Floating Dock (Active Conversation)**:
  * As soon as a query is submitted or a suggestion chip is clicked, the centered hero disappears and the search input smoothly transitions to the **fixed bottom floating dock** (`messages.length > 0`).
  * During active conversations, users continue typing from the bottom floating dock while the conversation history, timeline, and citations flow in the scrollable canvas above.
  * Clicking **New Chat** or **Clear conversation** immediately resets the interface and returns the search box to the centered hero position.

