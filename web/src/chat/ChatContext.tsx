import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ApiError, streamChat } from "../api/client";
import type { ChatEvent, ChatTurn, Source } from "../api/types";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  standalone?: string; // the question as it was actually searched for (a follow-up rewritten from the history)
  sources?: Source[];
  error?: string; // the request or the answer failed; shown as a warning, never as the model's words
  streaming?: boolean;
  scope?: string; // on a question: the file the search was limited to
}

export interface ChatScope {
  doc_id: string;
  filename: string;
}

interface ChatState {
  messages: ChatMessage[];
  busy: boolean;
  scope: ChatScope | null; // the one file the next questions search in; null means every document the user may read
  setScope: (scope: ChatScope | null) => void;
  send: (text: string) => Promise<void>;
  stop: () => void;
  clear: () => void;
}

const ChatContext = createContext<ChatState | null>(null);
const HISTORY_LIMIT = 12; // the server trims it again and is the authority

export function toHistory(messages: ChatMessage[], question: string): ChatTurn[] {
  const past = messages
    .filter((m) => !m.error && !m.streaming && m.content.trim() !== "")
    .slice(-HISTORY_LIMIT)
    .map((m) => ({ role: m.role, content: m.content }));
  return [...past, { role: "user", content: question }];
}

let counter = 0;
const newId = () => `m${++counter}`;

export function ChatProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [scope, setScope] = useState<ChatScope | null>(null);
  const scopeRef = useRef<ChatScope | null>(null);
  const controller = useRef<AbortController | null>(null);
  const messagesRef = useRef<ChatMessage[]>([]);
  useEffect(() => {
    messagesRef.current = messages; // `send` reads the latest conversation without being recreated on every token
    scopeRef.current = scope;
  }, [messages, scope]);

  const send = useCallback(async (text: string) => {
    const question = text.trim();
    if (!question) return;
    const history = toHistory(messagesRef.current, question);
    const chosen = scopeRef.current;
    const answerId = newId();
    setMessages((m) => [...m, { id: newId(), role: "user", content: question, scope: chosen?.filename }, { id: answerId, role: "assistant", content: "", streaming: true }]);
    setBusy(true);
    const abort = new AbortController();
    controller.current = abort;

    const patch = (fn: (m: ChatMessage) => ChatMessage) => setMessages((all) => all.map((m) => (m.id === answerId ? fn(m) : m)));
    const onEvent = (e: ChatEvent) => {
      if (e.type === "token") patch((m) => ({ ...m, content: m.content + e.text }));
      else if (e.type === "standalone") patch((m) => ({ ...m, standalone: e.question }));
      else if (e.type === "sources") patch((m) => ({ ...m, sources: e.sources }));
      else if (e.type === "error") patch((m) => ({ ...m, error: e.message }));
    };
    try {
      await streamChat(history, onEvent, abort.signal, chosen ? [chosen.doc_id] : undefined);
    } catch (err) {
      if (!(err instanceof DOMException && err.name === "AbortError")) {
        const message = err instanceof ApiError ? err.message : "Could not reach the server. Check your connection and try again.";
        patch((m) => ({ ...m, error: message }));
      }
    } finally {
      patch((m) => ({ ...m, streaming: false }));
      setBusy(false);
      controller.current = null;
    }
  }, []);

  const stop = useCallback(() => controller.current?.abort(), []);
  const clear = useCallback(() => {
    controller.current?.abort();
    setMessages([]);
    setScope(null);
  }, []);

  const value = useMemo(() => ({ messages, busy, scope, setScope, send, stop, clear }), [messages, busy, scope, send, stop, clear]);
  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat(): ChatState {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChat must be used inside <ChatProvider>");
  return ctx;
}
