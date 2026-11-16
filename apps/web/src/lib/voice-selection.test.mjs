import test from "node:test";
import assert from "node:assert/strict";

import {
  selectPreferredVoice,
  voicesForLocale,
} from "./voice-selection.mjs";

const voice = (name, lang, voiceURI = name) => ({
  name,
  lang,
  voiceURI,
  localService: true,
});

test("prefers a pleasant female-associated voice matching the requested locale", () => {
  const voices = [
    voice("Hindi Default", "hi-IN"),
    voice("Microsoft Swara Online (Natural)", "hi-IN"),
    voice("Samantha", "en-US"),
  ];

  assert.equal(
    selectPreferredVoice(voices, "hi-IN").voiceURI,
    "Microsoft Swara Online (Natural)",
  );
});

test("keeps a user-selected matching voice instead of replacing it", () => {
  const voices = [
    voice("Hindi Default", "hi-IN", "default-hi"),
    voice("Microsoft Swara", "hi-IN", "swara-hi"),
  ];

  assert.equal(
    selectPreferredVoice(voices, "hi-IN", "default-hi").voiceURI,
    "default-hi",
  );
});

test("falls back to the same base language before an unrelated voice", () => {
  const voices = [
    voice("Samantha", "en-US"),
    voice("Hindi UK", "hi-GB"),
  ];

  assert.deepEqual(voicesForLocale(voices, "hi-IN"), [voices[1]]);
  assert.equal(selectPreferredVoice(voices, "hi-IN").voiceURI, "Hindi UK");
});
