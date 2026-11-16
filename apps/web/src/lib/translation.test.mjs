import test from "node:test";
import assert from "node:assert/strict";

import {
  GoogleCloudTranslationProvider,
  googleClientOptionsFromEnvironment,
  splitProtectedSegments,
} from "./translation.mjs";

test("builds server-only Google credentials from deployment environment variables", () => {
  assert.deepEqual(
    googleClientOptionsFromEnvironment({
      GOOGLE_CLOUD_PROJECT_ID: "demo-project",
      GOOGLE_CLOUD_CLIENT_EMAIL: "translator@demo-project.iam.gserviceaccount.com",
      GOOGLE_CLOUD_PRIVATE_KEY: "first-line\\nsecond-line",
    }),
    {
      projectId: "demo-project",
      credentials: {
        client_email: "translator@demo-project.iam.gserviceaccount.com",
        private_key: "first-line\nsecond-line",
      },
    },
  );
});

test("supports JSON credentials string or object from deployment environment variables", () => {
  const jsonString = JSON.stringify({
    project_id: "demo-project",
    client_email: "translator@demo-project.iam.gserviceaccount.com",
    private_key: "-----BEGIN KEY-----\\nabc\\n-----END KEY-----",
  });
  assert.deepEqual(
    googleClientOptionsFromEnvironment({
      GOOGLE_APPLICATION_CREDENTIALS_JSON: jsonString,
    }),
    {
      projectId: "demo-project",
      credentials: {
        client_email: "translator@demo-project.iam.gserviceaccount.com",
        private_key: "-----BEGIN KEY-----\nabc\n-----END KEY-----",
      },
    },
  );
});

test("strips surrounding quotes from private key string", () => {
  assert.deepEqual(
    googleClientOptionsFromEnvironment({
      GOOGLE_CLOUD_PROJECT_ID: "demo-project",
      GOOGLE_CLOUD_CLIENT_EMAIL: "translator@demo-project.iam.gserviceaccount.com",
      GOOGLE_CLOUD_PRIVATE_KEY: '"-----BEGIN KEY-----\\nabc\\n-----END KEY-----"',
    }),
    {
      projectId: "demo-project",
      credentials: {
        client_email: "translator@demo-project.iam.gserviceaccount.com",
        private_key: "-----BEGIN KEY-----\nabc\n-----END KEY-----",
      },
    },
  );
});

test("uses Application Default Credentials when deployment variables are absent or incomplete", () => {
  assert.deepEqual(googleClientOptionsFromEnvironment({}), {});
  assert.deepEqual(
    googleClientOptionsFromEnvironment({
      GOOGLE_CLOUD_PROJECT_ID: "incomplete-project",
    }),
    {},
  );
});

test("keeps citations, statute names, locators, dates, URLs, and numeric ranges out of translated segments", () => {
  const text = "Under Trade Marks Rules, 2017, Rule 23 applies on 2026-09-16 [1] at https://example.test/source for pages 59–61.";
  const segments = splitProtectedSegments(text, [
    "Trade Marks Rules, 2017",
    "Rule 23",
  ]);
  const protectedText = segments
    .filter((part) => part.protected)
    .map((part) => part.text);

  assert.deepEqual(protectedText, [
    "Trade Marks Rules, 2017",
    "Rule 23",
    "2026-09-16",
    "[1]",
    "https://example.test/source",
    "59–61",
  ]);
});

test("translates only explanatory segments and reconstructs protected text byte for byte", async () => {
  const calls = [];
  const client = {
    async translate(parts, options) {
      calls.push({ parts, options });
      return [parts.map((part) => `हिन्दी:${part}`)];
    },
  };
  const provider = new GoogleCloudTranslationProvider({ client });

  const result = await provider.translate({
    text: "The application is signed under Trade Marks Rules, 2017 — Rule 23 [1].",
    sourceLanguage: "en",
    targetLanguage: "hi",
    protectedTerms: ["Trade Marks Rules, 2017", "Rule 23"],
  });

  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].options, {
    from: "en",
    to: "hi",
    format: "text",
  });
  assert.equal(
    calls[0].parts.join("").includes("Trade Marks Rules, 2017"),
    false,
  );
  assert.equal(result.includes("Trade Marks Rules, 2017 — Rule 23 [1]"), true);
});

test("returns English text unchanged without calling the external client", async () => {
  const client = {
    async translate() {
      throw new Error("must not be called");
    },
  };
  const provider = new GoogleCloudTranslationProvider({ client });

  const result = await provider.translate({
    text: "Already English [1].",
    sourceLanguage: "en",
    targetLanguage: "en",
  });

  assert.equal(result, "Already English [1].");
});

test("preserves whitespace beside protected legal terms when the provider trims it", async () => {
  const client = {
    async translate(parts) {
      return [parts.map((part) => part.trim())];
    },
  };
  const provider = new GoogleCloudTranslationProvider({ client });

  const result = await provider.translate({
    text: "Explanation before Trade Marks Rules, 2017 and after.",
    sourceLanguage: "en",
    targetLanguage: "hi",
    protectedTerms: ["Trade Marks Rules, 2017"],
  });

  assert.equal(
    result,
    "Explanation before Trade Marks Rules, 2017 and after.",
  );
});
