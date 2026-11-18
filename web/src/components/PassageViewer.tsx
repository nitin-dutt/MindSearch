import { useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, FileText, Loader2, ScanText, X } from "lucide-react";
import { ApiError, getPassages } from "../api/client";
import type { PassagePage } from "../api/types";

/** What to open: a document, and optionally the passage a citation points at (shown highlighted, with its neighbours). */
export interface ViewTarget {
  doc_id: string;
  filename: string;
  chunk_index: number | null;
}

const WINDOW = 3; // passages per screen: the cited one with one before and one after

/**
 * The "view source" panel. It shows a document's text AS IT WAS INDEXED, a few passages at a time: the server keeps only the
 * encrypted passages (the uploaded file is deleted after processing), so there is no original page to draw. The server decides
 * whether this user may read the document; a document that is gone or not allowed simply says so.
 */
export function PassageViewer({ target, onClose }: { target: ViewTarget | null; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (target && !dialog.open) dialog.showModal?.();
    if (!target && dialog.open) dialog.close?.();
  }, [target]);

  return (
    <dialog
      ref={ref}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      aria-labelledby="viewer-title"
      className="m-auto w-[min(44rem,calc(100vw-2rem))] rounded-2xl border border-line bg-panel p-0 text-ink shadow-2xl backdrop:bg-black/50"
    >
      {/* keyed by what is being viewed, so opening another source starts from a clean state at the right passage */}
      {target && <ViewerBody key={`${target.doc_id}:${target.chunk_index}`} target={target} onClose={onClose} />}
    </dialog>
  );
}

type Loaded = { start: number; page: PassagePage } | { start: number; error: string };

function ViewerBody({ target, onClose }: { target: ViewTarget; onClose: () => void }) {
  // Start at the passage before the cited one, so the citation is read in context.
  const [start, setStart] = useState(Math.max(0, (target.chunk_index ?? 0) - 1));
  const [loaded, setLoaded] = useState<Loaded | null>(null);

  useEffect(() => {
    let alive = true; // ignore an answer that arrives after the panel moved on or closed
    getPassages(target.doc_id, start, WINDOW)
      .then((page) => alive && setLoaded({ start, page }))
      .catch((e) => alive && setLoaded({ start, error: e instanceof ApiError && e.status === 404 ? "This document is no longer available to you." : "Could not load the passages. Try again." }));
    return () => {
      alive = false;
    };
  }, [target.doc_id, start]);

  const current = loaded?.start === start ? loaded : null; // an answer for an earlier window is not shown as this one
  const page = current && "page" in current ? current.page : null;
  const error = current && "error" in current ? current.error : null;
  const total = page?.total ?? 0;
  const last = page && page.passages.length > 0 ? page.passages[page.passages.length - 1].chunk_index : start;

  return (
    <div className="flex max-h-[85vh] flex-col">
      <div className="flex items-start gap-3 border-b border-line p-4">
        <FileText className="mt-0.5 size-5 shrink-0 text-accent" aria-hidden />
        <div className="min-w-0 flex-1">
          <h2 id="viewer-title" className="truncate text-base font-semibold" title={target.filename}>
            {target.filename}
          </h2>
          <p className="text-xs text-muted">The text as it was indexed. The uploaded file itself is not kept.</p>
        </div>
        <button onClick={onClose} aria-label="Close" className="rounded-lg p-1.5 text-muted hover:bg-accent-soft hover:text-ink">
          <X className="size-4" />
        </button>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto p-4" aria-live="polite">
        {error && (
          <div role="alert" className="rounded-xl border border-warn/40 bg-warn-soft px-3.5 py-2.5 text-sm text-ink">
            {error}
          </div>
        )}
        {current === null && (
          <div className="flex items-center gap-2 py-6 text-sm text-muted" role="status">
            <Loader2 className="size-4 animate-spin" aria-hidden /> Loading passages…
          </div>
        )}
        {page !== null && page.passages.length === 0 && <p className="py-6 text-sm text-muted">There is nothing more to show here.</p>}
        {page?.passages.map((p) => {
          const cited = p.chunk_index === target.chunk_index;
          return (
            <article key={p.chunk_index} data-passage={p.chunk_index} className={`rounded-xl border p-3.5 text-sm ${cited ? "border-accent/60 bg-accent-soft" : "border-line bg-bg"}`}>
              <div className="mb-1.5 flex flex-wrap items-center gap-2 text-xs text-muted">
                <span className="font-semibold">{`Passage ${p.chunk_index + 1}${total > 0 ? ` of ${total}` : ""}`}</span>
                {p.page !== null && <span>· page {p.page}</span>}
                {p.scanned && (
                  <span className="inline-flex items-center gap-1" title="This text was read from a scanned page or a picture (OCR), so it may contain reading errors.">
                    <ScanText className="size-3" aria-hidden /> scanned
                  </span>
                )}
                {cited && <span className="rounded-full bg-brand px-2 py-0.5 font-medium">cited passage</span>}
              </div>
              <p className="whitespace-pre-wrap leading-relaxed">{p.text}</p>
            </article>
          );
        })}
      </div>

      <div className="flex items-center justify-between gap-2 border-t border-line p-3">
        <button
          onClick={() => setStart((s) => Math.max(0, s - WINDOW))}
          disabled={start === 0 || page === null}
          className="inline-flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-sm text-muted hover:bg-accent-soft hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
        >
          <ChevronLeft className="size-4" aria-hidden /> Earlier
        </button>
        <button
          onClick={() => setStart((s) => s + WINDOW)}
          disabled={page === null || last + 1 >= total}
          className="inline-flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-sm text-muted hover:bg-accent-soft hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
        >
          Later <ChevronRight className="size-4" aria-hidden />
        </button>
      </div>
    </div>
  );
}
