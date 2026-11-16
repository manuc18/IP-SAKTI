const PREFERRED_VOICE_NAMES = [
  "swara",
  "heera",
  "aditi",
  "neerja",
  "lekha",
  "veena",
  "shruti",
  "pallavi",
  "dhwani",
  "sapna",
  "vaani",
  "tanishaa",
  "samantha",
  "ava",
  "aria",
  "jenny",
  "zira",
  "karen",
  "moira",
  "tessa",
  "female",
  "woman",
];

function normalizeLocale(locale) {
  return String(locale || "").replace("_", "-").toLowerCase();
}

export function voicesForLocale(voices, locale) {
  const requested = normalizeLocale(locale);
  const baseLanguage = requested.split("-")[0];
  const exact = voices.filter(
    (voice) => normalizeLocale(voice.lang) === requested,
  );
  if (exact.length > 0) return exact;
  return voices.filter(
    (voice) => normalizeLocale(voice.lang).split("-")[0] === baseLanguage,
  );
}

function preferenceScore(voice) {
  const name = String(voice.name || "").toLowerCase();
  const preferenceIndex = PREFERRED_VOICE_NAMES.findIndex((candidate) =>
    name.includes(candidate),
  );
  const preferredScore =
    preferenceIndex === -1 ? 0 : 1_000 - preferenceIndex;
  return preferredScore + (voice.localService ? 10 : 0);
}

export function selectPreferredVoice(voices, locale, selectedVoiceURI = "") {
  const matchingVoices = voicesForLocale(voices, locale);
  const selected = matchingVoices.find(
    (voice) => voice.voiceURI === selectedVoiceURI,
  );
  if (selected) return selected;
  if (matchingVoices.length === 0) return null;

  return [...matchingVoices].sort(
    (left, right) => preferenceScore(right) - preferenceScore(left),
  )[0];
}
