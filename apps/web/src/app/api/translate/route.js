import {
  GoogleCloudTranslationProvider,
  googleClientOptionsFromEnvironment,
} from "../../../lib/translation.mjs";
import { createTranslationHandler } from "../../../lib/translation-route.mjs";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const handler = createTranslationHandler({
  provider: new GoogleCloudTranslationProvider(),
});

export const POST = handler;

export async function GET() {
  const options = googleClientOptionsFromEnvironment(process.env);
  const hasOptions = Boolean(
    options.projectId &&
      options.credentials?.client_email &&
      options.credentials?.private_key,
  );

  let testStatus = "credentials_not_found";
  let errorMessage = null;

  if (hasOptions) {
    try {
      const provider = new GoogleCloudTranslationProvider();
      const res = await provider.translate({
        text: "Hello",
        sourceLanguage: "en",
        targetLanguage: "hi",
      });
      testStatus = res ? "ok" : "empty_response";
    } catch (err) {
      testStatus = "error";
      errorMessage = err?.message || String(err);
    }
  }

  return new Response(
    JSON.stringify(
      {
        configured: hasOptions,
        testStatus,
        errorMessage,
        detectedVars: {
          GOOGLE_APPLICATION_CREDENTIALS_JSON: Boolean(
            process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON,
          ),
          GOOGLE_APPLICATION_CREDENTIALS: Boolean(
            process.env.GOOGLE_APPLICATION_CREDENTIALS,
          ),
          GOOGLE_CLOUD_PROJECT_ID: Boolean(process.env.GOOGLE_CLOUD_PROJECT_ID),
          GOOGLE_CLOUD_CLIENT_EMAIL: Boolean(
            process.env.GOOGLE_CLOUD_CLIENT_EMAIL,
          ),
          GOOGLE_CLOUD_PRIVATE_KEY: Boolean(
            process.env.GOOGLE_CLOUD_PRIVATE_KEY,
          ),
        },
        projectId: options.projectId || null,
        clientEmail: options.credentials?.client_email || null,
        hasPrivateKey: Boolean(options.credentials?.private_key),
      },
      null,
      2,
    ),
    {
      status: 200,
      headers: { "Content-Type": "application/json" },
    },
  );
}
