export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function POST(request) {
  try {
    const body = await request.json();
    const question = body?.question || body?.message || body?.prompt;
    if (!question || typeof question !== "string") {
      return new Response(JSON.stringify({ title: "New Conversation" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }

    const apiKey =
      process.env.GROQ_API_KEY ||
      process.env.NEXT_PUBLIC_GROQ_API_KEY;

    if (!apiKey) {
      const fallbackTitle =
        question.slice(0, 30).trim() + (question.length > 30 ? "..." : "");
      return new Response(JSON.stringify({ title: fallbackTitle }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }

    const language = body?.language || "English";

    const systemPrompt =
      language && language !== "English"
        ? `You are a concise title generator. Generate a 3-5 word title summarizing the user query for a sidebar chat list. The title MUST be written in ${language}. Return ONLY the title text without quotes, markdown, or punctuation at the end.`
        : "You are a concise title generator. Generate a 3-5 word title summarizing the user query for a sidebar chat list. Return ONLY the title text without quotes, markdown, or punctuation at the end.";

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
            content: systemPrompt,
          },
          { role: "user", content: question },
        ],
        temperature: 0.2,
        max_tokens: 150,
      }),
    });

    if (res.ok) {
      const data = await res.json();
      let title = data.choices?.[0]?.message?.content?.trim();
      if (title) {
        title = title.replace(/^["']|["']$/g, "").replace(/\.$/, "").trim();
        return new Response(JSON.stringify({ title }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
    }

    const fallbackTitle =
      question.slice(0, 32).trim() + (question.length > 32 ? "..." : "");
    return new Response(JSON.stringify({ title: fallbackTitle }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    return new Response(JSON.stringify({ title: "Legal Consultation" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }
}
