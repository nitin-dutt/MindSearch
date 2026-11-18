import { parseSse } from "./sse";

const frame = (o: unknown) => `data: ${JSON.stringify(o)}\n\n`;

describe("parseSse", () => {
  it("reads tokens, the rewritten question, sources, errors and the end marker", () => {
    const text =
      frame({ type: "standalone", question: "What is the price?" }) +
      frame({ type: "token", choices: [{ delta: { content: "Hello " } }] }) +
      frame({ type: "error", message: "interrupted" }) +
      frame({ type: "outcome", outcome: "answered" }) +
      frame({ type: "sources", sources: [{ n: 1, doc_id: "d", filename: "a.pdf", chunk_index: 0, page: 3, cited: true }] }) +
      "data: [DONE]\n\n";
    const { events, rest } = parseSse(text);
    expect(rest).toBe("");
    expect(events.map((e) => e.type)).toEqual(["standalone", "token", "error", "outcome", "sources", "done"]);
    expect(events[1]).toEqual({ type: "token", text: "Hello " });
  });

  it("reads the list of searches the agent strategy ran, keeping only text", () => {
    const { events } = parseSse(frame({ type: "searches", queries: ["refund policy", 7, "warranty terms"] }));
    expect(events).toEqual([{ type: "searches", queries: ["refund policy", "warranty terms"] }]);
  });

  it("keeps an unfinished event for the next network chunk", () => {
    const whole = frame({ type: "token", choices: [{ delta: { content: "split me" } }] });
    const first = parseSse(whole.slice(0, 20));
    expect(first.events).toEqual([]);
    const second = parseSse(first.rest + whole.slice(20));
    expect(second.events).toEqual([{ type: "token", text: "split me" }]);
  });

  it("skips malformed frames and empty tokens instead of failing", () => {
    const { events } = parseSse("data: {not json\n\n" + frame({ type: "token", choices: [{ delta: { content: "" } }] }) + frame({ type: "token", choices: [{ delta: { content: "ok" } }] }));
    expect(events).toEqual([{ type: "token", text: "ok" }]);
  });

  it("skips a frame whose type it does not know, an outcome it does not know, and an untyped frame", () => {
    const { events } = parseSse(frame({ type: "from_the_future", x: 1 }) + frame({ type: "outcome", outcome: "maybe" }) + frame({ choices: [{ delta: { content: "untyped" } }] }));
    expect(events).toEqual([]);
  });

  it("ignores lines that are not data lines", () => {
    expect(parseSse(": comment\nevent: x\n\n").events).toEqual([]);
  });
});
