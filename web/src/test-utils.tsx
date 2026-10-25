import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";
import App from "./App";
import { setToken } from "./api/client";
import { AuthProvider } from "./auth/AuthContext";
import { ToastProvider } from "./components/Toast";

type Handler = (req: { body: BodyInit | null | undefined; headers: Headers }) => Response | object | Promise<Response | object>;

export const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

/** A streaming response made of the given raw text chunks (so tests can split an event across chunks). */
export function sse(chunks: string[], status = 200) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      chunks.forEach((c) => controller.enqueue(encoder.encode(c)));
      controller.close();
    },
  });
  return new Response(stream, { status, headers: { "Content-Type": "text/event-stream" } });
}

export const event = (obj: unknown) => `data: ${JSON.stringify(obj)}\n\n`;
export const tokenEvent = (text: string) => event({ choices: [{ delta: { content: text } }] });

/** Replaces fetch with handlers keyed by "METHOD /path" (the /api prefix is stripped). Unmatched requests fail the test loudly. */
export function mockApi(handlers: Record<string, Handler>) {
  const calls: { key: string; body: BodyInit | null | undefined }[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const path = String(input).replace(/^\/api/, "");
    const key = `${(init.method ?? "GET").toUpperCase()} ${path}`;
    calls.push({ key, body: init.body });
    const handler = handlers[key];
    if (!handler) throw new Error(`Unexpected API call: ${key}`);
    const out = await handler({ body: init.body, headers: new Headers(init.headers) });
    return out instanceof Response ? out : json(out);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock };
}

export const ME = { username: "alice", level: "Manager", groups: ["HR"], is_super: false };

export function renderApp(route = "/", opts: { signedIn?: boolean } = { signedIn: true }) {
  if (opts.signedIn !== false) setToken("test-token");
  return render(
    <MemoryRouter initialEntries={[route]}>
      <ToastProvider>
        <AuthProvider>
          <App />
        </AuthProvider>
      </ToastProvider>
    </MemoryRouter>,
  );
}
