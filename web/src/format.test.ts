import { answerKind, formatBytes, timeAgo } from "./format";

describe("answerKind", () => {
  it("recognises the I-don't-know sentence, also with a typographic apostrophe", () => {
    expect(answerKind("I don't know based on the documents available to you.")).toBe("refusal");
    expect(answerKind("I don’t know based on the documents available to you.")).toBe("refusal");
  });
  it("recognises the nothing-found message", () => {
    expect(answerKind("I couldn't find any relevant information in the documents available to you. If you expected...")).toBe("none_found");
  });
  it("treats everything else, including text that merely mentions not knowing, as an answer", () => {
    expect(answerKind("The report says nobody knows the price [1].")).toBe("answer");
    expect(answerKind("")).toBe("answer");
  });
});

describe("formatting", () => {
  it("formats sizes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
  });
  it("describes recent times relatively and old ones as dates", () => {
    const now = new Date("2026-10-17T12:00:00Z");
    expect(timeAgo("2026-10-17T11:59:50Z", now)).toBe("just now");
    expect(timeAgo("2026-10-17T11:30:00Z", now)).toMatch(/30 minutes ago/);
    expect(timeAgo("2026-10-17T07:00:00Z", now)).toMatch(/5 hours ago/);
    expect(timeAgo("2026-01-02T00:00:00Z", now)).toMatch(/2026/);
    expect(timeAgo("garbage", now)).toBe("");
  });
});
