import { parseSse } from "./sse";
import type { ChatEvent, ChatTurn, DocumentEntry, Me, Policy, UploadResult } from "./types";

const BASE = "/api"; // nginx (production) and the Vite dev server both forward /api to the FastAPI service

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/* --- the access token ------------------------------------------------------------------------------------------------
 * Kept in memory, and in sessionStorage so a page reload does not log you out. Never localStorage (it outlives the tab) and never
 * in a URL. If the API says 401 the token is dropped and the app is told, so it can show the login screen. */
const TOKEN_KEY = "mindsearch.token";
let token: string | null = null;
let onUnauthorized: () => void = () => {};

export function loadToken(): string | null {
  if (token) return token;
  try {
    token = sessionStorage.getItem(TOKEN_KEY);
  } catch {
    token = null;
  }
  return token;
}

export function setToken(value: string | null) {
  token = value;
  try {
    if (value) sessionStorage.setItem(TOKEN_KEY, value);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable (private window): the token simply lives in memory */
  }
}

export function setUnauthorizedHandler(handler: () => void) {
  onUnauthorized = handler;
}

async function errorFrom(res: Response): Promise<ApiError> {
  let detail = `Request failed (${res.status})`;
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") detail = body.detail;
    else if (Array.isArray(body?.detail) && body.detail[0]?.msg) detail = String(body.detail[0].msg);
  } catch {
    /* not JSON: keep the generic message */
  }
  return new ApiError(res.status, detail);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const t = loadToken();
  if (t) headers.set("Authorization", `Bearer ${t}`);
  const res = await fetch(`${BASE}${path}`, { ...init, headers });
  if (res.status === 401 && path !== "/v1/auth/login") {
    setToken(null);
    onUnauthorized();
  }
  if (!res.ok) throw await errorFrom(res);
  return (await res.json()) as T;
}

export async function login(username: string, password: string): Promise<void> {
  const res = await request<{ access_token: string }>("/v1/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  setToken(res.access_token);
}

export const getMe = () => request<Me>("/v1/me");
export const getPolicy = () => request<Policy>("/v1/policy");
export const listDocuments = async () => (await request<{ documents: DocumentEntry[] }>("/v1/documents")).documents;
export const deleteDocument = (docId: string) => request<{ status: string }>(`/v1/documents/${docId}`, { method: "DELETE" });

export async function uploadFiles(files: File[], level: string, groups: string[]): Promise<UploadResult[]> {
  const form = new FormData();
  form.set("session_id", crypto.randomUUID());
  form.set("level", level);
  form.set("groups", groups.join(","));
  for (const f of files) form.append("files", f);
  const res = await request<{ documents: UploadResult[] }>("/v1/ingest", { method: "POST", body: form });
  return res.documents;
}

/**
 * Ask a question and stream the answer. `history` is the conversation so far INCLUDING the new question (the server trims it).
 * Resolves when the stream ends; rejects with ApiError if the request itself fails (for example HTTP 503 when the model is down).
 */
export async function streamChat(history: ChatTurn[], onEvent: (e: ChatEvent) => void, signal?: AbortSignal, docIds?: string[]): Promise<void> {
  const headers = new Headers({ "Content-Type": "application/json" });
  const t = loadToken();
  if (t) headers.set("Authorization", `Bearer ${t}`);
  const res = await fetch(`${BASE}/v1/chat/completions`, {
    method: "POST",
    headers,
    body: JSON.stringify({ model: "rag-model", stream: true, messages: history, ...(docIds && docIds.length > 0 ? { doc_ids: docIds } : {}) }),
    signal,
  });
  if (res.status === 401) {
    setToken(null);
    onUnauthorized();
  }
  if (!res.ok || !res.body) throw await errorFrom(res);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const { events, rest } = parseSse(buffer);
    buffer = rest;
    events.forEach(onEvent);
  }
  buffer += "\n\n"; // flush a final event that arrived without its blank line
  parseSse(buffer).events.forEach(onEvent);
}
