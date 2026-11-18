import { formatBytes, timeAgo } from "./format";

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
