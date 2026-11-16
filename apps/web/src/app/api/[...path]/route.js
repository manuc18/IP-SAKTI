/**
 * Next.js catch-all API proxy route
 * Forwards /api/* requests to the Python backend on Render.
 * Replaces the Express gateway — no separate server needed on Vercel.
 *
 * Env var required (set in Vercel dashboard):
 *   PYTHON_BACKEND_URL=https://ip-sakti-o2fn.onrender.com
 */

const BACKEND = process.env.PYTHON_BACKEND_URL || "http://localhost:8000";

export async function GET(request, { params }) {
  const path = (await params).path.join("/");
  const url = `${BACKEND}/api/${path}`;
  const res = await fetch(url, { cache: "no-store" });
  const data = await res.json();
  return Response.json(data, { status: res.status });
}

export async function POST(request, { params }) {
  const path = (await params).path.join("/");
  const url = `${BACKEND}/api/${path}`;
  const body = await request.text();
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    cache: "no-store",
  });
  const data = await res.json();
  return Response.json(data, { status: res.status });
}
