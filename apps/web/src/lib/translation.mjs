import { v2 as translateV2 } from "@google-cloud/translate";

export const SUPPORTED_LANGUAGE_CODES = new Set([
  "en",
  "hi",
  "sa",
  "mr",
  "ta",
  "te",
  "bn",
  "gu",
  "kn",
  "pa",
]);

export function googleClientOptionsFromEnvironment(env = {}) {
  const jsonCreds =
    env.GOOGLE_APPLICATION_CREDENTIALS_JSON ||
    env.GOOGLE_CREDENTIALS ||
    env.GOOGLE_CLOUD_CREDENTIALS ||
    (env.GOOGLE_APPLICATION_CREDENTIALS?.trim().startsWith("{")
      ? env.GOOGLE_APPLICATION_CREDENTIALS
      : null);

  if (jsonCreds) {
    try {
      const parsed =
        typeof jsonCreds === "string" ? JSON.parse(jsonCreds) : jsonCreds;
      if (parsed.project_id && parsed.client_email && parsed.private_key) {
        let privateKey = parsed.private_key;
        if (typeof privateKey === "string") {
          privateKey = privateKey.trim();
          if (
            (privateKey.startsWith('"') && privateKey.endsWith('"')) ||
            (privateKey.startsWith("'") && privateKey.endsWith("'"))
          ) {
            privateKey = privateKey.slice(1, -1);
          }
          privateKey = privateKey.replace(/\\n/g, "\n");
        }
        return {
          projectId: parsed.project_id,
          credentials: {
            client_email: parsed.client_email,
            private_key: privateKey,
          },
        };
      }
    } catch {
      // Fall through to individual environment variables
    }
  }

  const projectId = (
    env.GOOGLE_CLOUD_PROJECT_ID ||
    env.GOOGLE_PROJECT_ID ||
    env.GCP_PROJECT_ID ||
    env.PROJECT_ID
  )?.trim();

  const clientEmail = (
    env.GOOGLE_CLOUD_CLIENT_EMAIL ||
    env.GOOGLE_CLIENT_EMAIL ||
    env.GCP_CLIENT_EMAIL ||
    env.CLIENT_EMAIL
  )?.trim();

  let privateKey =
    env.GOOGLE_CLOUD_PRIVATE_KEY ||
    env.GOOGLE_PRIVATE_KEY ||
    env.GCP_PRIVATE_KEY ||
    env.PRIVATE_KEY;

  if (!projectId || !clientEmail || !privateKey) return {};

  if (typeof privateKey === "string") {
    privateKey = privateKey.trim();
    if (
      (privateKey.startsWith('"') && privateKey.endsWith('"')) ||
      (privateKey.startsWith("'") && privateKey.endsWith("'"))
    ) {
      privateKey = privateKey.slice(1, -1);
    }
    privateKey = privateKey.replace(/\\n/g, "\n");
  }

  return {
    projectId,
    credentials: {
      client_email: clientEmail,
      private_key: privateKey,
    },
  };
}

const BUILT_IN_PROTECTED_PATTERNS = [
  /\[\d+\]/gu,
  /https?:\/\/[^\s<>"]+/gu,
  /\b\d{4}-\d{2}-\d{2}\b/gu,
  /\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b/gu,
  /\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b/giu,
  /\b(?:Section|Article|Rule|Regulation|Schedule|Clause|Form)\s+[A-Za-z0-9()./-]+/giu,
  /\b\d+\s*[–—-]\s*\d+\b/gu,
];

const URL_TRAILING_PUNCTUATION = /[.,!?;:]+$/u;

function collectExactTermRanges(text, protectedTerms) {
  const ranges = [];
  const terms = [...new Set(protectedTerms)]
    .filter((term) => typeof term === "string" && term.length > 0)
    .sort((left, right) => right.length - left.length);

  for (const term of terms) {
    let fromIndex = 0;
    while (fromIndex < text.length) {
      const start = text.indexOf(term, fromIndex);
      if (start === -1) break;
      ranges.push({ start, end: start + term.length });
      fromIndex = start + term.length;
    }
  }
  return ranges;
}

function collectPatternRanges(text) {
  const ranges = [];
  for (const pattern of BUILT_IN_PROTECTED_PATTERNS) {
    pattern.lastIndex = 0;
    let match;
    while ((match = pattern.exec(text)) !== null) {
      let value = match[0];
      if (value.startsWith("http")) {
        value = value.replace(URL_TRAILING_PUNCTUATION, "");
      }
      if (value) {
        ranges.push({ start: match.index, end: match.index + value.length });
      }
      if (match[0].length === 0) pattern.lastIndex += 1;
    }
  }
  return ranges;
}

function mergeRanges(ranges) {
  const ordered = ranges.sort(
    (left, right) => left.start - right.start || right.end - left.end,
  );
  const merged = [];

  for (const range of ordered) {
    const previous = merged.at(-1);
    if (!previous || range.start >= previous.end) {
      merged.push({ ...range });
    } else if (range.end > previous.end) {
      previous.end = range.end;
    }
  }
  return merged;
}

export function splitProtectedSegments(text, protectedTerms = []) {
  if (typeof text !== "string" || text.length === 0) return [];

  const ranges = mergeRanges([
    ...collectExactTermRanges(text, protectedTerms),
    ...collectPatternRanges(text),
  ]);
  if (ranges.length === 0) return [{ text, protected: false }];

  const segments = [];
  let cursor = 0;
  for (const range of ranges) {
    if (range.start > cursor) {
      segments.push({ text: text.slice(cursor, range.start), protected: false });
    }
    segments.push({ text: text.slice(range.start, range.end), protected: true });
    cursor = range.end;
  }
  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor), protected: false });
  }
  return segments;
}

async function translateViaGtx(text, from, to) {
  const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=${encodeURIComponent(from)}&tl=${encodeURIComponent(to)}&dt=t&q=${encodeURIComponent(text)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GTX translation returned status ${res.status}`);
  const data = await res.json();
  if (Array.isArray(data?.[0])) {
    const result = data[0].map((item) => item?.[0] || "").join("");
    if (result) return result;
  }
  throw new Error("Invalid GTX response format");
}

async function translateViaGroq(text, from, to, apiKey) {
  const languageNames = {
    en: "English",
    hi: "Hindi",
    sa: "Sanskrit",
    mr: "Marathi",
    ta: "Tamil",
    te: "Telugu",
    bn: "Bengali",
    gu: "Gujarati",
    kn: "Kannada",
    pa: "Punjabi",
  };
  const fromName = languageNames[from] || from;
  const toName = languageNames[to] || to;

  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "openai/gpt-oss-120b",
      messages: [
        {
          role: "system",
          content: `You are a professional legal and technical translator. Translate the given text from ${fromName} to ${toName}. Return ONLY the direct translation without explanation, notes, or quotation marks.`,
        },
        { role: "user", content: text },
      ],
      temperature: 0.1,
    }),
  });
  if (!res.ok) throw new Error(`Groq translation returned status ${res.status}`);
  const data = await res.json();
  const content = data.choices?.[0]?.message?.content?.trim();
  if (!content) throw new Error("Empty response from Groq translation");
  return content;
}

async function fallbackTranslatePart(text, from, to, env) {
  try {
    return await translateViaGtx(text, from, to);
  } catch (gtxErr) {
    const groqKey = env?.GROQ_API_KEY || process.env.GROQ_API_KEY;
    if (groqKey) {
      try {
        return await translateViaGroq(text, from, to, groqKey);
      } catch (groqErr) {
        throw new Error(
          `Fallback translation failed (GTX: ${gtxErr?.message}; Groq: ${groqErr?.message})`,
        );
      }
    }
    throw gtxErr;
  }
}

let googleClientCooldownUntil = 0;
const GOOGLE_CLIENT_COOLDOWN_MS = 5 * 60 * 1000;

export class GoogleCloudTranslationProvider {
  constructor({ client, env } = {}) {
    this.customClient = client;
    this.env = env;
  }

  getClient() {
    if (this.customClient) return this.customClient;
    const currentEnv = this.env || process.env;
    return new translateV2.Translate(
      googleClientOptionsFromEnvironment(currentEnv),
    );
  }

  async translate({
    text,
    sourceLanguage,
    targetLanguage,
    protectedTerms = [],
  }) {
    if (sourceLanguage === targetLanguage) return text;

    const segments = splitProtectedSegments(text, protectedTerms);
    const translatableIndexes = segments.flatMap((segment, index) =>
      !segment.protected && /\p{L}/u.test(segment.text) ? [index] : [],
    );
    if (translatableIndexes.length === 0) return text;

    const translatableParts = translatableIndexes.map((index) => {
      const match = segments[index].text.match(/^(\s*)([\s\S]*?)(\s*)$/u);
      return {
        leading: match[1],
        text: match[2],
        trailing: match[3],
      };
    });
    const translatableText = translatableParts.map((part) => part.text);

    let translatedText;
    const client = this.getClient();
    const shouldSkipGoogle =
      !this.customClient && Date.now() < googleClientCooldownUntil;

    if (!shouldSkipGoogle) {
      try {
        const [translated] = await client.translate(translatableText, {
          from: sourceLanguage,
          to: targetLanguage,
          format: "text",
        });
        translatedText = Array.isArray(translated) ? translated : [translated];
        if (translatedText.length !== translatableIndexes.length) {
          throw new Error("Translation provider returned an incomplete response.");
        }
      } catch (primaryErr) {
        if (this.customClient) {
          throw primaryErr;
        }
        googleClientCooldownUntil = Date.now() + GOOGLE_CLIENT_COOLDOWN_MS;
      }
    }

    if (!translatedText) {
      const currentEnv = this.env || process.env;
      translatedText = await Promise.all(
        translatableText.map((part) =>
          fallbackTranslatePart(part, sourceLanguage, targetLanguage, currentEnv),
        ),
      );
    }

    translatableIndexes.forEach((segmentIndex, translatedIndex) => {
      const boundary = translatableParts[translatedIndex];
      segments[segmentIndex] = {
        text: `${boundary.leading}${translatedText[translatedIndex]}${boundary.trailing}`,
        protected: false,
      };
    });
    return segments.map((segment) => segment.text).join("");
  }
}
