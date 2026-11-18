import type { ChatEvent, ReplyKind, Source } from "./types";

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

const REPLY_KINDS: readonly ReplyKind[] = ["answered", "refused", "none_found"];

/** One frame as an event. The server names each frame's `type`; a frame of an unknown type (a newer server) is skipped. */
function toEvent(data: string): ChatEvent | null {
  let obj: unknown;
  try {
    obj = JSON.parse(data);
  } catch {
    return null; // a malformed frame is skipped, never fatal
  }
  if (typeof obj !== "object" || obj === null) return null;
  const o = obj as Record<string, unknown>;
  switch (o.type) {
    case "token": {
      // token frames keep the OpenAI-style shape: choices[0].delta.content
      const delta = Array.isArray(o.choices) ? (o.choices[0] as { delta?: { content?: unknown } } | undefined)?.delta : undefined;
      return typeof delta?.content === "string" && delta.content !== "" ? { type: "token", text: delta.content } : null;
    }
    case "outcome":
      return REPLY_KINDS.includes(o.outcome as ReplyKind) ? { type: "outcome", outcome: o.outcome as ReplyKind } : null;
    case "standalone":
      return typeof o.question === "string" ? { type: "standalone", question: o.question } : null;
    case "searches":
      return Array.isArray(o.queries) ? { type: "searches", queries: o.queries.filter((q): q is string => typeof q === "string") } : null;
    case "sources":
      return Array.isArray(o.sources) ? { type: "sources", sources: o.sources as Source[] } : null;
    case "error":
      return typeof o.message === "string" ? { type: "error", message: o.message } : null;
    default:
      return null;
  }
}
