import { defaultDevUser, parseDevUsers } from "./devAuth";

const RAW = JSON.stringify([
  { label: "Employee", username: "e", password: "p1" },
  { label: "Manager", username: "m", password: "p2" },
]);

describe("development login shortcut", () => {
  it("is off unless it is a dev run AND the flag is 1 AND accounts are given", () => {
    expect(parseDevUsers(false, "1", RAW)).toEqual([]); // a production build
    expect(parseDevUsers(true, undefined, RAW)).toEqual([]); // flag missing
    expect(parseDevUsers(true, "0", RAW)).toEqual([]);
    expect(parseDevUsers(true, "1", undefined)).toEqual([]);
    expect(parseDevUsers(true, "1", RAW)).toHaveLength(2);
  });

  it("ignores malformed values and accounts missing a field instead of throwing", () => {
    expect(parseDevUsers(true, "1", "not json")).toEqual([]);
    expect(parseDevUsers(true, "1", '{"a":1}')).toEqual([]);
    expect(parseDevUsers(true, "1", JSON.stringify([{ username: "x" }, { label: "L", username: "u", password: "p" }]))).toHaveLength(1);
  });

  it("starts as the wanted account, else the first one, else none", () => {
    const users = parseDevUsers(true, "1", RAW);
    expect(defaultDevUser(users, "m")?.username).toBe("m");
    expect(defaultDevUser(users, "unknown")?.username).toBe("e");
    expect(defaultDevUser([], "m")).toBeNull();
  });
});
