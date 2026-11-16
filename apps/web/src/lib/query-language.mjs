// Which language a typed query must be translated FROM before retrieval.
//
// Retrieval is English-only. The dropdown used to be the only signal, so a
// Hindi query typed while the dropdown said English reached the backend as raw
// Devanagari, matched no English regime term and came back as a clarification
// ("No answer was published"). The script of the text is a stronger signal than
// the dropdown, and it is deterministic: no model, no network.

const SCRIPT_LANGUAGES = [
  [/\p{Script=Devanagari}/u, "hi"],
  [/\p{Script=Bengali}/u, "bn"],
  [/\p{Script=Gurmukhi}/u, "pa"],
  [/\p{Script=Gujarati}/u, "gu"],
  [/\p{Script=Tamil}/u, "ta"],
  [/\p{Script=Telugu}/u, "te"],
  [/\p{Script=Kannada}/u, "kn"],
];

// Languages written in Devanagari. When the dropdown already names one of them
// it is more specific than the script, so it wins.
const DEVANAGARI_LANGUAGES = new Set(["hi", "mr", "sa"]);

export function detectScriptLanguage(text) {
  if (typeof text !== "string") return null;
  for (const [pattern, code] of SCRIPT_LANGUAGES) {
    if (pattern.test(text)) return code;
  }
  return null;
}

// Mixed input ("मैं अपनी firm को trademark कैसे कराऊँ?") is detected by its
// Indic script and translated whole; the translator keeps the English legal
// terms, which is what retrieval needs. Pure Latin text keeps the dropdown's
// language, exactly as before.
export function resolveQueryLanguage(text, selectedCode = "en") {
  const script = detectScriptLanguage(text);
  if (!script) return selectedCode;
  if (script === "hi" && DEVANAGARI_LANGUAGES.has(selectedCode)) return selectedCode;
  if (script === selectedCode) return selectedCode;
  return script;
}

// The translation client falls back to the original text when every provider
// fails. For an Indic-script query that fallback would silently send the raw
// script to an English-only retriever, so it is reported as a failure instead.
export function queryTranslationFailed(original, translated) {
  if (!detectScriptLanguage(original)) return false;
  return (
    typeof translated !== "string" ||
    !translated.trim() ||
    translated.trim() === original.trim()
  );
}
