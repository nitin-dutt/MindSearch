import { parseSse } from "./sse";

const frame = (o: unknown) => `data: ${JSON.stringify(o)}\n\n`;

describe("parseSse", () => {
  it("reads tokens, the rewritten question, sources, errors and the end marker", () => {
    const text =
      frame({ standalone_question: "What is the price?" }) +
      frame({ choices: [{ delta: { content: "Hello " } }] }) +
      frame({ error: "interrupted" }) +
      frame({ sources: [{ n: 1, doc_id: "d", filename: "a.pdf", chunk_index: 0, page: 3, cited: true }] }) +
      "data: [DONE]\n\n";
    const { events, rest } = parseSse(text);
    expect(rest).toBe("");
    expect(events.map((e) => e.type)).toEqual(["standalone", "token", "error", "sources", "done"]);
    expect(events[1]).toEqual({ type: "token", text: "Hello " });
  });

  it("keeps an unfinished event for the next network chunk", () => {
    const whole = frame({ choices: [{ delta: { content: "split me" } }] });
    const first = parseSse(whole.slice(0, 20));
    expect(first.events).toEqual([]);
    const second = parseSse(first.rest + whole.slice(20));
    expect(second.events).toEqual([{ type: "token", text: "split me" }]);
  });

  it("skips malformed frames and empty tokens instead of failing", () => {
    const { events } = parseSse("data: {not json\n\n" + frame({ choices: [{ delta: { content: "" } }] }) + frame({ choices: [{ delta: { content: "ok" } }] }));
    expect(events).toEqual([{ type: "token", text: "ok" }]);
  });

  it("ignores lines that are not data lines", () => {
    expect(parseSse(": comment\nevent: x\n\n").events).toEqual([]);
  });
});
