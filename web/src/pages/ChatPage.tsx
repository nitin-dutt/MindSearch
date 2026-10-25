import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { AlertTriangle, ArrowUp, ChevronDown, FileText, HelpCircle, Lock, Quote, Search, ShieldCheck, Square, Trash2 } from "lucide-react";
import { listDocuments } from "../api/client";
import { useChat, type ChatMessage, type ChatScope } from "../chat/ChatContext";
import { MindMark } from "../components/MindMark";
import { RichText } from "../components/RichText";
import { answerKind } from "../format";
import type { Source } from "../api/types";

export function ChatPage() {
  const { messages, busy, scope, setScope, send, stop, clear } = useChat();
  const [draft, setDraft] = useState("");
  const [docs, setDocs] = useState<ChatScope[]>([]);
  const bottom = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);

  // Starter prompts come from the user's own documents. Generic questions ("what are the main topics?") match nothing
  // in a search-based system and would only end in "I don't know", so these just start a sentence for the user to finish.
  useEffect(() => {
    let alive = true;
    listDocuments()
      .then((all) => alive && setDocs(all.filter((d) => d.status === "done").map((d) => ({ doc_id: d.doc_id, filename: d.filename }))))
      .catch(() => {}); // starters are optional; the page works without them
    return () => {
      alive = false;
    };
  }, []);

  function startWith(doc: ChatScope) {
    setScope(doc);
    setDraft("What does this document say about ");
    box.current?.focus();
  }

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [messages]);

  // Grow the box with its text (up to its max height) and shrink it again after sending. It also re-measures when the box changes width,
  // because a measurement taken before the styles have loaded (or before a resize) would otherwise leave it far too tall.
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const fit = () => {
      el.style.height = "auto";
      el.style.height = `${el.scrollHeight}px`;
    };
    fit();
    if (typeof ResizeObserver === "undefined") return;
    let lastWidth = el.clientWidth;
    const watcher = new ResizeObserver(() => {
      if (el.clientWidth !== lastWidth) {
        lastWidth = el.clientWidth;
        fit();
      }
    });
    watcher.observe(el);
    return () => watcher.disconnect();
  }, [draft]);

  function submit() {
    const text = draft.trim();
    if (!text || busy) return;
    setDraft("");
    void send(text);
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="mx-auto flex w-full max-w-3xl items-center justify-between px-4 pt-4">
        <h1 className="text-lg font-semibold">Chat</h1>
        {messages.length > 0 && (
          <button onClick={clear} className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm text-muted hover:bg-accent-soft hover:text-ink">
            <Trash2 className="size-4" aria-hidden /> New chat
          </button>
        )}
      </div>

      {/* The scroll area spans the full width so its scrollbar sits at the window edge; the text stays in a centred column. */}
      <div className="flex-1 overflow-y-auto" role="log" aria-live="polite" aria-label="Conversation">
       <div className="mx-auto max-w-3xl space-y-6 px-4 py-6">
        {messages.length === 0 ? (
          <div className="pt-10">
            <div className="flex flex-col items-center text-center">
              <MindMark className="size-20" />
              <h2 className="mt-5 text-3xl font-semibold tracking-tight">Ask about your documents</h2>
              <p className="mt-2 max-w-md text-sm text-muted">Answers come only from documents your account may read, with numbered sources you can check.</p>
              <div className="mt-5 flex flex-wrap justify-center gap-2 text-xs text-muted">
                {[
                  [Lock, "Private and offline"],
                  [Quote, "Cited answers"],
                  [ShieldCheck, "Role-based access"],
                ].map(([Icon, label]) => (
                  <span key={label as string} className="inline-flex items-center gap-1.5 rounded-full border border-line bg-panel px-3 py-1 shadow-card">
                    <Icon className="size-3.5 text-accent" aria-hidden />
                    {label as string}
                  </span>
                ))}
              </div>
            </div>
            {docs.length > 0 && (
              <div className="mt-8">
                <div className="mb-2 text-center text-xs font-semibold uppercase tracking-wide text-muted">Start with one of your documents</div>
                <div className="flex flex-wrap justify-center gap-2">
                  {docs.slice(0, 3).map((doc) => (
                    <button
                      key={doc.doc_id}
                      onClick={() => startWith(doc)}
                      className="flex max-w-full items-center gap-2 rounded-full border border-line bg-panel px-3 py-1.5 text-sm text-muted shadow-card transition hover:-translate-y-0.5 hover:border-accent hover:text-ink"
                    >
                      <FileText className="size-3.5 shrink-0" aria-hidden />
                      <span className="truncate">{doc.filename}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          messages.map((m) => <Message key={m.id} message={m} />)
        )}
        <div ref={bottom} />
       </div>
      </div>

      <div className="border-t border-line bg-bg px-4 pb-4 pt-3">
       <div className="mx-auto max-w-3xl">
        {docs.length > 0 && (
          <label className="mb-2 flex items-center gap-2 text-xs text-muted">
            <FileText className="size-3.5 shrink-0" aria-hidden />
            Search in
            <select
              value={scope?.doc_id ?? ""}
              onChange={(e) => setScope(docs.find((d) => d.doc_id === e.target.value) ?? null)}
              className="min-w-0 max-w-[60%] truncate rounded-lg border border-line bg-panel px-2 py-1 text-xs text-ink"
              aria-label="Search in"
            >
              <option value="">All my documents</option>
              {docs.map((d) => (
                <option key={d.doc_id} value={d.doc_id}>
                  {d.filename}
                </option>
              ))}
            </select>
          </label>
        )}
        <div className="flex items-end gap-2 rounded-2xl border border-line bg-panel py-1.5 pl-3 pr-1.5 shadow-sm transition focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/25">
          <textarea
            ref={box}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKey}
            rows={1}
            placeholder="Ask a question about your documents…"
            aria-label="Your question"
            className="max-h-40 min-h-9 flex-1 resize-none bg-transparent py-1.5 text-sm outline-none focus-visible:outline-none"
          />
          {busy ? (
            <button
              onClick={stop}
              aria-label="Stop generating"
              title="Stop generating"
              className="relative grid size-9 shrink-0 place-items-center rounded-xl bg-accent-soft text-accent transition hover:brightness-110"
            >
              <span className="absolute inset-1 animate-spin rounded-full border-2 border-accent/25 border-t-accent" aria-hidden />
              <Square className="size-3 fill-current" aria-hidden />
            </button>
          ) : (
            <button
              onClick={submit}
              disabled={!draft.trim()}
              aria-label="Send"
              title="Send"
              className={`grid size-9 shrink-0 place-items-center rounded-xl transition ${draft.trim() ? "bg-brand hover:brightness-95" : "bg-line text-muted"}`}
            >
              <ArrowUp className="size-5" aria-hidden />
            </button>
          )}
        </div>
        <p className="mt-2 text-center text-xs text-muted">Enter to send · Shift+Enter for a new line</p>
       </div>
      </div>
    </div>
  );
}

function SourceRow({ s }: { s: Source }) {
  const where = s.page !== null ? `page ${s.page}` : s.chunk_index !== null ? `part ${s.chunk_index + 1}` : "";
  return (
    <li
      data-source={s.n}
      className={`flex items-center gap-2 rounded-lg border px-2 py-1.5 text-xs ${s.cited ? "border-accent/40 bg-panel" : "border-line bg-panel text-muted"}`}
    >
      <span className={`grid size-5 shrink-0 place-items-center rounded-md font-semibold ${s.cited ? "bg-brand" : "bg-line text-muted"}`}>{s.n}</span>
      <FileText className="size-3.5 shrink-0 text-muted" aria-hidden />
      <span className="min-w-0 truncate font-medium text-ink">{s.filename}</span>
      {where && <span className="shrink-0 text-muted">· {where}</span>}
    </li>
  );
}

// Collapsed by default so an answer reads cleanly; clicking a [n] citation (or the header) opens it. Sources the answer actually used are
// listed first; the rest are tucked away so the same file is not shown over and over.
function SourceList({ sources, open, onToggle }: { sources: Source[]; open: boolean; onToggle: () => void }) {
  const cited = sources.filter((s) => s.cited);
  const rest = sources.filter((s) => !s.cited);
  const label = cited.length > 0 ? `Sources used (${cited.length})` : "Sources searched";
  return (
    <div className="space-y-1.5 pt-1">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="flex items-center gap-1 text-xs font-semibold uppercase tracking-wide text-muted hover:text-ink"
      >
        <ChevronDown className={`size-3.5 transition-transform ${open ? "" : "-rotate-90"}`} aria-hidden />
        {label}
      </button>
      {open && cited.length > 0 && (
        <ul className="grid gap-1.5 sm:grid-cols-2">
          {cited.map((s) => (
            <SourceRow key={s.n} s={s} />
          ))}
        </ul>
      )}
      {open && rest.length > 0 && (
        <details className="group text-xs text-muted">
          <summary className="cursor-pointer select-none py-1 hover:text-ink">
            {cited.length > 0 ? `${rest.length} more passage${rest.length === 1 ? "" : "s"} looked at, not used` : "Passages looked at"}
          </summary>
          <ul className="mt-1 grid gap-1.5 sm:grid-cols-2">
            {rest.map((s) => (
              <SourceRow key={s.n} s={s} />
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function Message({ message }: { message: ChatMessage }) {
  const ref = useRef<HTMLDivElement>(null);
  const [showSources, setShowSources] = useState(false);
  const [jump, setJump] = useState<{ n: number; tick: number } | null>(null);

  // After a citation click has opened the list, scroll to the source and flash it.
  useEffect(() => {
    if (!jump) return;
    const el = ref.current?.querySelector<HTMLElement>(`[data-source="${jump.n}"]`);
    el?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
    el?.classList.remove("flash");
    void el?.offsetWidth; // restart the highlight animation if it already ran
    el?.classList.add("flash");
  }, [jump]);

  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%]">
          <div className="whitespace-pre-wrap rounded-2xl rounded-br-md bg-brand px-4 py-2.5 text-sm">{message.content}</div>
          {message.scope && <div className="mt-1 truncate text-right text-xs text-muted">in {message.scope}</div>}
        </div>
      </div>
    );
  }

  const kind = answerKind(message.content);
  const jumpToSource = (n: number) => {
    setShowSources(true);
    setJump((j) => ({ n, tick: (j?.tick ?? 0) + 1 }));
  };

  return (
    <div ref={ref} className="flex gap-3">
      <MindMark className="mt-1 size-8" />
      <div className={`min-w-0 flex-1 space-y-3 ${kind === "answer" ? "rounded-2xl rounded-tl-md border border-line bg-panel p-4 shadow-card" : ""}`}>
        {message.standalone && (
          <div className="inline-flex items-center gap-1.5 rounded-full bg-accent-soft px-2.5 py-1 text-xs text-accent">
            <Search className="size-3" aria-hidden /> Searched for: {message.standalone}
          </div>
        )}

        {message.streaming && message.content === "" && !message.error && (
          <div className="flex items-center gap-2 text-sm text-muted" role="status">
            <span className="flex gap-1" aria-hidden>
              <i className="size-1.5 animate-bounce rounded-full bg-muted [animation-delay:-0.3s]" />
              <i className="size-1.5 animate-bounce rounded-full bg-muted [animation-delay:-0.15s]" />
              <i className="size-1.5 animate-bounce rounded-full bg-muted" />
            </span>
            Searching your documents…
          </div>
        )}

        {message.content !== "" &&
          (kind === "answer" ? (
            <div className="text-sm">
              <RichText text={message.content} sources={message.sources} onCite={jumpToSource} />
            </div>
          ) : (
            <div className="flex items-start gap-3 rounded-xl border border-line bg-panel p-3.5 text-sm">
              <HelpCircle className="mt-0.5 size-4 shrink-0 text-muted" aria-hidden />
              <div>
                <p>{message.content}</p>
                <p className="mt-1 text-xs text-muted">
                  {kind === "refusal"
                    ? "The documents available to you do not contain this, so nothing was guessed."
                    : "If you expected an answer, check that the document finished processing and that your access level covers it."}
                </p>
              </div>
            </div>
          ))}

        {message.error && (
          <div role="alert" className="flex items-start gap-2 rounded-xl border border-warn/40 bg-warn-soft px-3.5 py-2.5 text-sm text-warn">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
            <span className="text-ink">{message.error}</span>
          </div>
        )}

        {message.sources && message.sources.length > 0 && <SourceList sources={message.sources} open={showSources} onToggle={() => setShowSources((o) => !o)} />}
      </div>
    </div>
  );
}
