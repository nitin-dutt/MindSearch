import type { ChatEvent, Source } from "./types";

/**
 * The chat endpoint streams Server-Sent Events: lines like `data: {...}` separated by a blank line, ending with `data: [DONE]`.
 * Network chunks do not respect event boundaries, so this keeps the unfinished tail and returns it for the next chunk.
 */
export function parseSse(buffer: string): { events: ChatEvent[]; rest: string } {
  const events: ChatEvent[] = [];
  const frames = buffer.split("\n\n");
  const rest = frames.pop() ?? ""; // the last piece may be incomplete
  for (const frame of frames) {
    for (const line of frame.split("\n")) {
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") {
        events.push({ type: "done" });
        continue;
      }
      const event = toEvent(data);
      if (event) events.push(event);
    }
  }
  return { events, rest };
}

function toEvent(data: string): ChatEvent | null {
  let obj: unknown;
  try {
    obj = JSON.parse(data);
  } catch {
    return null; // a malformed frame is skipped, never fatal
  }
  if (typeof obj !== "object" || obj === null) return null;
  const o = obj as Record<string, unknown>;
  if (Array.isArray(o.choices)) {
    const delta = (o.choices[0] as { delta?: { content?: unknown } } | undefined)?.delta;
    return typeof delta?.content === "string" && delta.content !== "" ? { type: "token", text: delta.content } : null;
  }
  if (typeof o.standalone_question === "string") return { type: "standalone", question: o.standalone_question };
  if (Array.isArray(o.sources)) return { type: "sources", sources: o.sources as Source[] };
  if (typeof o.error === "string") return { type: "error", message: o.error };
  return null;
}
