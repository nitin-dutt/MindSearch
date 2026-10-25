import { ApiError, listDocuments, login, loadToken, setToken, setUnauthorizedHandler, streamChat, uploadFiles } from "./client";
import { event, json, mockApi, sse, tokenEvent } from "../test-utils";
import type { ChatEvent } from "./types";

afterEach(() => vi.unstubAllGlobals());

describe("api client", () => {
  it("logs in with JSON and then sends the token on later calls", async () => {
    const { calls, fetchMock } = mockApi({
      "POST /v1/auth/login": () => ({ access_token: "tok-1", token_type: "bearer", expires_in: 3600 }),
      "GET /v1/documents": () => ({ documents: [] }),
    });
    await login("alice", "pw");
    expect(JSON.parse(String(calls[0].body))).toEqual({ username: "alice", password: "pw" });
    await listDocuments();
    const headers = new Headers((fetchMock.mock.calls[1][1] as RequestInit).headers);
    expect(headers.get("Authorization")).toBe("Bearer tok-1");
    expect(sessionStorage.getItem("mindsearch.token")).toBe("tok-1");
    expect(localStorage.getItem("mindsearch.token")).toBeNull(); // never in localStorage
  });

  it("drops the token and calls the handler on a 401 (but not for a failed login)", async () => {
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    setToken("old");
    mockApi({
      "GET /v1/documents": () => json({ detail: "Not authenticated" }, 401),
      "POST /v1/auth/login": () => json({ detail: "Incorrect username or password" }, 401),
    });
    await expect(listDocuments()).rejects.toMatchObject({ status: 401 });
    expect(handler).toHaveBeenCalledTimes(1);
    expect(loadToken()).toBeNull();
    await expect(login("a", "b")).rejects.toBeInstanceOf(ApiError);
    expect(handler).toHaveBeenCalledTimes(1); // the login failure did not look like an expired session
  });

  it("uploads files as multipart form data with the chosen level and departments", async () => {
    const { calls } = mockApi({ "POST /v1/ingest": () => ({ documents: [{ filename: "a.txt", doc_id: "x", status: "queued", duplicate: false }] }) });
    const file = new File(["hello"], "a.txt", { type: "text/plain" });
    const out = await uploadFiles([file], "Manager", ["HR", "Finance"]);
    const form = calls[0].body as FormData;
    expect(form.get("level")).toBe("Manager");
    expect(form.get("groups")).toBe("HR,Finance");
    expect((form.getAll("files")[0] as File).name).toBe("a.txt");
    expect(out[0].status).toBe("queued");
  });

  it("surfaces the server's message for a failed request", async () => {
    mockApi({ "POST /v1/ingest": () => json({ detail: "You can only upload documents at an access level you could read yourself." }, 403) });
    await expect(uploadFiles([new File(["x"], "a.txt")], "Executive", [])).rejects.toThrow("access level you could read yourself");
  });

  it("streams chat events even when one is split across network chunks", async () => {
    const full = tokenEvent("Hel") + tokenEvent("lo") + "data: [DONE]\n\n";
    mockApi({ "POST /v1/chat/completions": () => sse([full.slice(0, 15), full.slice(15)]) });
    const got: ChatEvent[] = [];
    await streamChat([{ role: "user", content: "hi" }], (e) => got.push(e));
    expect(got.map((e) => e.type)).toEqual(["token", "token", "done"]);
  });

  it("rejects with the server's message when the model is unavailable (HTTP 503)", async () => {
    mockApi({ "POST /v1/chat/completions": () => json({ detail: "The answer service is unavailable. Please try again in a moment." }, 503) });
    await expect(streamChat([{ role: "user", content: "hi" }], () => {})).rejects.toMatchObject({ status: 503 });
  });

  it("passes an error event through to the caller", async () => {
    mockApi({ "POST /v1/chat/completions": () => sse([tokenEvent("Par"), event({ error: "The answer was interrupted. Please try again." }), "data: [DONE]\n\n"]) });
    const got: ChatEvent[] = [];
    await streamChat([{ role: "user", content: "hi" }], (e) => got.push(e));
    expect(got.find((e) => e.type === "error")).toBeTruthy();
  });
});
