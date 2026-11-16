import {
  googleClientOptionsFromEnvironment,
} from "../../../lib/translation.mjs";
import { v2 as translateV2 } from "@google-cloud/translate";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const MAX_TEXTS = 100;
const MAX_TEXT_LENGTH = 8_000;
const TOTAL_CHAR_LIMIT = 50_000;

const SUPPORTED = new Set([
  "en", "hi", "sa", "mr", "ta", "te", "bn", "gu", "kn", "pa",
]);

let sharedClient = null;
const GOOGLE_CLIENT_COOLDOWN_MS = 10 * 60 * 1000;

function getClient() {
  if (sharedClient) return sharedClient;
  sharedClient = new translateV2.Translate(
    googleClientOptionsFromEnvironment(process.env),
  );
  return sharedClient;
}

async function singleViaGtx(text, from, to) {
  const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=${encodeURIComponent(from)}&tl=${encodeURIComponent(to)}&dt=t&q=${encodeURIComponent(text)}`;
  const res = await fetch(url, {
    headers: { "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)" },
  });
  if (!res.ok) throw new Error(`GTX ${res.status}`);
  const data = await res.json();
  if (Array.isArray(data?.[0])) {
    return data[0].map((item) => item?.[0] || "").join("");
  }
  throw new Error("Invalid GTX response");
}

const LANGUAGE_NAMES = {
  en: "English", hi: "Hindi", sa: "Sanskrit", mr: "Marathi",
  ta: "Tamil", te: "Telugu", bn: "Bengali", gu: "Gujarati",
  kn: "Kannada", pa: "Punjabi",
};

async function batchViaGroq(texts, from, to, apiKey) {
  const fromName = LANGUAGE_NAMES[from] || from;
  const toName = LANGUAGE_NAMES[to] || to;
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
          content: `You are a professional legal translator. Translate the JSON array of strings from ${fromName} to ${toName}. Return ONLY a valid JSON array of translated strings matching the input array length and order exactly. Do not output markdown code fences, notes, or explanations.`,
        },
        { role: "user", content: JSON.stringify(texts) },
      ],
      temperature: 0.1,
    }),
  });

  if (!res.ok) throw new Error(`Groq ${res.status}`);
  const data = await res.json();
  const raw = data.choices?.[0]?.message?.content?.trim() || "";
  const cleaned = raw.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "").trim();
  const parsed = JSON.parse(cleaned);
  if (Array.isArray(parsed) && parsed.length === texts.length) {
    return parsed.map((item, idx) => (typeof item === "string" && item.trim() ? item.trim() : texts[idx]));
  }
  throw new Error("Invalid array from Groq");
}

async function singleViaGroq(text, from, to, apiKey) {
  const fromName = LANGUAGE_NAMES[from] || from;
  const toName = LANGUAGE_NAMES[to] || to;
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
          content: `You are a professional legal translator. Translate the text from ${fromName} to ${toName}. Return ONLY the direct translation without explanation, notes, or extra quotes.`,
        },
        { role: "user", content: text },
      ],
      temperature: 0.1,
    }),
  });
  if (!res.ok) throw new Error(`Groq ${res.status}`);
  const data = await res.json();
  return data.choices?.[0]?.message?.content?.trim() || "";
}

async function batchFallback(texts, from, to) {
  const groqKey = process.env.GROQ_API_KEY;

  // 1. Try batch Groq first (fast, single round-trip, translates entire array at once)
  if (groqKey) {
    try {
      return await batchViaGroq(texts, from, to, groqKey);
    } catch {
      // Fall through to per-item fallback
    }
  }

  // 2. Per-item GTX / Groq fallback
  return Promise.all(
    texts.map(async (text) => {
      try {
        return await singleViaGtx(text, from, to);
      } catch {
        if (groqKey) {
          try {
            return await singleViaGroq(text, from, to, groqKey);
          } catch {
            // Keep original
          }
        }
        return text;
      }
    }),
  );
}

export async function POST(request) {
  let body;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const { texts, sourceLanguage, targetLanguage } = body ?? {};

  if (!Array.isArray(texts) || texts.length === 0) {
    return Response.json({ error: "texts must be a non-empty array" }, { status: 400 });
  }
  if (texts.length > MAX_TEXTS) {
    return Response.json({ error: `Too many texts (max ${MAX_TEXTS})` }, { status: 400 });
  }
  if (!SUPPORTED.has(sourceLanguage) || !SUPPORTED.has(targetLanguage)) {
    return Response.json({ error: "Unsupported language code" }, { status: 400 });
  }

  if (sourceLanguage === targetLanguage) {
    return Response.json({ translatedTexts: texts });
  }

  const translatedTexts = [...texts];
  const indexesToTranslate = [];
  const segmentTexts = [];
  let totalChars = 0;

  for (let i = 0; i < texts.length; i++) {
    const t = texts[i];
    if (!t || typeof t !== "string" || !t.trim()) continue;
    const truncated = t.length > MAX_TEXT_LENGTH ? t.slice(0, MAX_TEXT_LENGTH) : t;
    totalChars += truncated.length;
    if (totalChars > TOTAL_CHAR_LIMIT) break;
    indexesToTranslate.push(i);
    segmentTexts.push(truncated);
  }

  if (indexesToTranslate.length === 0) {
    return Response.json({ translatedTexts });
  }

  const now = Date.now();
  const shouldTryGoogle = !global.__googleClientCooldownUntil || now >= global.__googleClientCooldownUntil;

  if (shouldTryGoogle) {
    try {
      const client = getClient();
      const googleTranslatePromise = client.translate(segmentTexts, {
        from: sourceLanguage,
        to: targetLanguage,
        format: "text",
      });
      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error("Google Cloud translation timed out")), 1500),
      );
      const [results] = await Promise.race([googleTranslatePromise, timeoutPromise]);
      const resultsArr = Array.isArray(results) ? results : [results];
      indexesToTranslate.forEach((origIdx, i) => {
        if (resultsArr[i]) translatedTexts[origIdx] = resultsArr[i];
      });
      return Response.json({ translatedTexts });
    } catch {
      global.__googleClientCooldownUntil = Date.now() + GOOGLE_CLIENT_COOLDOWN_MS;
    }
  }

  // Fast Fallback: Groq batch translation
  try {
    const fallbackResults = await batchFallback(segmentTexts, sourceLanguage, targetLanguage);
    indexesToTranslate.forEach((origIdx, i) => {
      if (fallbackResults[i]) translatedTexts[origIdx] = fallbackResults[i];
    });
  } catch {
    // Graceful preservation
  }

  return Response.json({ translatedTexts });
}
