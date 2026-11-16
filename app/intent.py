"""A deterministic gate for inputs that are not questions about law.

"hi" is not a legal question, and the honest response to it is "hello" — not
"which area of IP law does your question concern?", and certainly not five
retrieved passages of the Patents Rules with an abstention attached. Before this
module existed every greeting went through `app/clarify.py`, which is the wrong
instrument: clarification exists to narrow a legal question that named no body
of law, and "thanks" is not an under-specified legal question.

"help" and "what can you do" are the same problem asked the other way round.
They are questions *about the assistant*, so the corpus has nothing to retrieve
against and the pipeline returned five weakly-matching passages and an
abstention — a user's first input answered with "no answer was published". They
are handled here as :data:`CAPABILITY`, under the same guards and the same
`SOCIAL` status as a greeting.

The whole gate is a closed set of phrases matched against the *entire* input.
No model, no network, no embedding, no scoring threshold. That is a deliberate
constraint rather than an optimisation: the failure this gate could cause —
answering a real legal question with "Hello! How can I help you?" — is worse
than the failure it fixes, so it is built so that failure is structurally
impossible rather than merely unlikely.

Three guards, in order, and each one alone is sufficient to refuse:

**Anchored inputs are never social.** If `app/clarify.anchors` finds a regime
term, a document alias or a provision identifier, the input is a legal question
and this module returns `None` without consulting its phrase list at all. This
is the same lexicon the router retrieves with, so anything retrievable is
un-greetable by construction.

**Every segment must be social.** The input is split on sentence punctuation and
each segment matched independently. "hi" is social. "hi, how do I register a
trade mark?" is not, because its second segment is not in the list — so the
common case of a polite preamble before a real question reaches retrieval
untouched.

**Only exact phrases match.** Not "contains", not "starts with", not a fuzzy
distance. `PHRASES` is a literal table, and a segment is social only when its
normalised form is a key in it. "What is the best pizza topping in Naples?" is
off-topic, not social: it is a question, it gets retrieved against, and it
abstains — which is the correct and already-tested behaviour, and this module
must not take it over.

What the gate returns is not an answer. `status` is `SOCIAL`, `answer` is null
and `evidence` is empty, so nothing downstream can mistake a greeting for a
published legal claim, and no grounded-answer or abstention metric counts it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.clarify import anchors

INTENT_VERSION = "0.1.0"

GREETING = "GREETING"
CAPABILITY = "CAPABILITY"
SMALL_TALK = "SMALL_TALK"
THANKS = "THANKS"
ACKNOWLEDGEMENT = "ACKNOWLEDGEMENT"
FAREWELL = "FAREWELL"

INTENTS = (GREETING, CAPABILITY, SMALL_TALK, THANKS, ACKNOWLEDGEMENT, FAREWELL)

#: One reply per intent. Written out rather than generated, because a fixed
#: string cannot hallucinate and every word here is reviewable. Each one ends by
#: handing the turn back to the subject the system actually knows about.
REPLIES: dict[str, str] = {
    GREETING: "Hello! I'm IP-SAKTI, a statutory-law assistant. Ask me a "
              "question about the legal instruments in my corpus and I'll "
              "retrieve the relevant provisions with citations.",
    CAPABILITY: "I answer questions from a fixed corpus of 22 Indian IP and "
                "regulatory-law instruments, and show the source citations for "
                "every answer. If the corpus doesn't support an answer, I say "
                "so. Ask me something like \"What is the term of a patent in "
                "India?\"",
    SMALL_TALK: "I'm doing well, thank you. What IP question can I help you with?",
    THANKS: "You're welcome.",
    ACKNOWLEDGEMENT: "Happy to help. Ask me another question whenever you're ready.",
    FAREWELL: "Goodbye — come back whenever you have another IP question.",
}

#: The closed table. Keys are *normalised* segments (see :func:`normalise`):
#: lower-cased, stripped of punctuation and non-letters, with runs of three or
#: more identical characters collapsed, so "Hiii!!!" and "HI" are both "hi".
#:
#: Entries are added only for a form a person would plausibly type on its own,
#: and only when that form carries no legal meaning in this corpus. "mark",
#: "act", "file" and "term" are absent for the obvious reason; "fine" and "good"
#: were dropped after being written, because a fine is a penalty and good faith
#: is a doctrine, and a table that swallows either is a table that can eat a
#: real query. Ambiguity resolves against inclusion, every time.
PHRASES: dict[str, str] = {}


def _register(intent: str, *phrases: str) -> None:
    for phrase in phrases:
        PHRASES[phrase] = intent


_register(
    GREETING,
    "hi", "hii", "hey", "helo", "hello", "heya", "hiya", "yo", "howdy",
    "greetings", "namaste", "namaskar", "hi there", "hey there",
    "hello there", "hi all", "hello all", "hi everyone", "hello everyone",
    "good morning", "good afternoon", "good evening", "good day",
    "morning", "gm", "hi hello", "hello hi",
)

#: Questions about the assistant rather than about law. They name no provision
#: and no regime, so retrieval has nothing to match and used to return five
#: passages and an abstention — the screenshot behaviour this intent removes.
#: "help" is safe as a whole-segment key for the same reason every other entry
#: is: "help me with section 3" anchors on a provision and never reaches here.
_register(
    CAPABILITY,
    "help", "help me", "i need help", "can you help", "can you help me",
    "can u help", "can you help me out", "please help", "need help",
    "how can you help", "how can you help me", "how can u help",
    "how can you help me today", "what can you do", "what do you do",
    "what can you do for me", "what can i ask", "what can i ask you",
    "what can you help with", "what can you help me with",
    "who are you", "what are you", "who r u", "who are u",
    "how do you work", "how does this work", "how does it work",
    "how do i use this", "how to use this", "what is ip sakti",
    "what is ip shakti", "tell me what you can do", "your capabilities",
)

_register(
    SMALL_TALK,
    "how are you", "how are you doing", "how are u", "how r u", "how r you",
    "hows it going", "how is it going", "hows everything", "how do you do",
    "whats up", "wassup", "sup", "you good", "are you there", "you there",
    "how you doing", "hope you are doing well", "hope you are well",
)

_register(
    THANKS,
    "thanks", "thank you", "thank u", "thanx", "thankyou", "thx", "ty",
    "thanks a lot", "thanks a ton", "thank you very much", "thank you so much",
    "many thanks", "much appreciated", "appreciate it", "cheers",
    "thanks so much", "thanks for the help", "thank you for the help",
)

_register(
    ACKNOWLEDGEMENT,
    "ok", "okay", "okey", "k", "kk", "cool", "nice", "great", "awesome",
    "perfect", "got it", "understood", "alright", "all right", "sure",
    "yay", "wow", "amazing", "excellent", "very good",
    "makes sense", "noted", "fair enough",
)

_register(
    FAREWELL,
    "bye", "byebye", "bye bye", "goodbye", "good bye", "see you", "see ya",
    "cya", "catch you later", "later", "good night", "goodnight", "gn",
    "take care",
)

#: Sentence punctuation and the separators people actually type between two
#: short utterances. A segment boundary is where "hi, thanks" becomes two
#: phrases and "hi, how do I register a trade mark?" becomes one social segment
#: and one that is not.
_SEGMENT_SPLIT = re.compile(r"[.!?,;:\n\r]+|\s+-+\s+")

#: Everything that is not a letter, a digit or a space is discarded before
#: matching, which is what lets "Hi!!! :)" and "hi" be the same key. Digits are
#: kept rather than dropped so that "section 3" can never normalise to a bare
#: word that might collide with a phrase.
_KEEP = re.compile(r"[^a-z0-9 ]+")

#: Three or more of the same character collapse to one: "heyyyy" -> "hey",
#: "hiiii" -> "hi", "yaaay" -> "yay". Two is left alone so "hello", "all" and
#: "good" survive unchanged.
_STRETCH = re.compile(r"(.)\1{2,}")


def normalise(segment: str) -> str:
    """The form matched against :data:`PHRASES`."""
    text = (segment or "").lower()
    # Drop the apostrophe rather than splitting on it, so "what's" becomes
    # "whats" and "hows" — the forms the table stores.
    text = text.replace("’", "").replace("‘", "").replace("'", "")
    text = _KEEP.sub(" ", text)
    text = _STRETCH.sub(r"\1", text)
    return " ".join(text.split())


@dataclass(frozen=True)
class SocialIntent:
    """A conversational input the pipeline should answer without retrieving."""

    intent: str
    reply: str
    matched: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "reply": self.reply,
            "matched_segments": list(self.matched),
            "deterministic": True,
            "uses_model": False,
            "retrieval_performed": False,
            "version": INTENT_VERSION,
        }


def classify(query: str,
             aliases: Mapping[str, Sequence[str]] | None = None
             ) -> SocialIntent | None:
    """The social intent of `query`, or `None` if it is anything else.

    `None` is the answer for every legal question, every off-topic question and
    every input the table does not list verbatim. Returning `None` costs a
    normal pipeline run; returning a `SocialIntent` for a legal question would
    cost the user their answer, so every ambiguous case resolves to `None`.
    """
    text = (query or "").strip()
    if not text:
        return None

    # Guard one. Anything the router could retrieve against is a legal question,
    # whatever else it also contains. "hi, section 3" is not a greeting.
    if anchors(text, aliases)["anchored"]:
        return None

    segments = [normalise(part) for part in _SEGMENT_SPLIT.split(text)]
    segments = [segment for segment in segments if segment]
    if not segments:
        return None

    # Guard two. Every segment, not the first and not the longest. One
    # unmatched segment means the input contains something this table has no
    # opinion about, and the pipeline is what forms opinions.
    found: list[str] = []
    for segment in segments:
        intent = PHRASES.get(segment)
        if intent is None:
            return None
        found.append(intent)

    # A mixed input takes the intent of its last segment, which is the one the
    # user finished on: "hi, thanks" is a thanks. Precedence is not scored —
    # position decides, deterministically.
    intent = found[-1]
    return SocialIntent(intent=intent, reply=REPLIES[intent],
                        matched=tuple(segments))


def intent_config() -> dict[str, Any]:
    return {
        "version": INTENT_VERSION,
        "deterministic": True,
        "uses_model": False,
        "intents": list(INTENTS),
        "phrase_count": len(PHRASES),
        "guarantee": "a social reply is returned only when the input anchors on "
                     "no regime, document or provision AND every one of its "
                     "segments is a verbatim entry in a fixed phrase table; it "
                     "carries status SOCIAL, no answer and no evidence, so it "
                     "can never be counted as a published legal answer",
    }
