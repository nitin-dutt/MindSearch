import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { AlertCircle, CheckCircle2, Clock, FileText, Loader2, Search, Trash2, Upload, Users } from "lucide-react";
import { deleteDocument, listDocuments } from "../api/client";
import type { DocStatus, DocumentEntry } from "../api/types";
import { Badge, ConfirmDialog, EmptyState, Skeleton } from "../components/ui";
import { useToast } from "../components/Toast";
import { formatBytes, timeAgo } from "../format";

export function StatusBadge({ status }: { status: DocStatus }) {
  if (status === "done")
    return (
      <Badge tone="ok">
        <CheckCircle2 className="size-3" aria-hidden /> Ready
      </Badge>
    );
  if (status === "failed")
    return (
      <Badge tone="danger">
        <AlertCircle className="size-3" aria-hidden /> Failed
      </Badge>
    );
  if (status === "processing")
    return (
      <Badge tone="warn">
        <Loader2 className="size-3 animate-spin" aria-hidden /> Processing
      </Badge>
    );
  return (
    <Badge tone="neutral">
      <Clock className="size-3" aria-hidden /> Queued
    </Badge>
  );
}

/** The document list, refreshed every few seconds while anything is still being processed. */
export function useDocuments() {
  const [docs, setDocs] = useState<DocumentEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setDocs(await listDocuments());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load documents.");
    }
  }, []);

  useEffect(() => {
    let alive = true; // ignore the answer if the page was left before it arrived
    listDocuments()
      .then((d) => alive && (setDocs(d), setError(null)))
      .catch((e) => alive && setError(e instanceof Error ? e.message : "Could not load documents."));
    return () => {
      alive = false;
    };
  }, []);

  const pending = docs?.some((d) => d.status === "queued" || d.status === "processing") ?? false;
  useEffect(() => {
    if (!pending) return;
    const id = setInterval(() => void refresh(), 3000);
    return () => clearInterval(id);
  }, [pending, refresh]);

  return { docs, error, refresh };
}

export function DocumentsPage() {
  const toast = useToast();
  const { docs, error, refresh } = useDocuments();
  const [query, setQuery] = useState("");
  const [toDelete, setToDelete] = useState<DocumentEntry | null>(null);
  const [deleting, setDeleting] = useState(false);

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (docs ?? []).filter((d) => !q || d.filename.toLowerCase().includes(q) || d.level.toLowerCase().includes(q) || d.groups.some((g) => g.toLowerCase().includes(q)));
  }, [docs, query]);

  async function confirmDelete() {
    if (!toDelete) return;
    setDeleting(true);
    try {
      await deleteDocument(toDelete.doc_id);
      toast.success(`Deleted ${toDelete.filename}`);
      setToDelete(null);
      await refresh();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Could not delete the document.");
    } finally {
      setDeleting(false);
    }
  }

  // The API only tells you who uploaded a document when it is yours (or you are an administrator): the same rule that allows deleting.
  const canDelete = (d: DocumentEntry) => d.can_delete;

  return (
    <div className="mx-auto max-w-5xl space-y-5 p-4 md:p-8">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold">Documents</h1>
          <p className="text-sm text-muted">Everything your account is allowed to read, with who else can read it.</p>
        </div>
        <Link to="/upload" className="inline-flex items-center gap-2 rounded-lg bg-accent px-3.5 py-2 text-sm font-medium text-accent-ink hover:opacity-90">
          <Upload className="size-4" aria-hidden /> Upload
        </Link>
      </div>

      {docs !== null && docs.length > 0 && (
        <label className="relative block">
          <span className="sr-only">Filter documents</span>
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted" aria-hidden />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter by name, level or department…"
            className="w-full rounded-xl border border-line bg-panel py-2.5 pl-9 pr-3 text-sm"
          />
        </label>
      )}

      {error && (
        <div role="alert" className="rounded-xl border border-danger/30 bg-danger-soft px-4 py-3 text-sm text-danger">
          {error}
        </div>
      )}

      {docs === null && !error && (
        <div className="space-y-2" aria-label="Loading documents">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-16" />
          ))}
        </div>
      )}

      {docs !== null && docs.length === 0 && (
        <EmptyState icon={<FileText className="size-6" />} title="No documents yet">
          Upload a PDF, Word, text or Markdown file and choose who may read it. It becomes searchable once processing finishes.
        </EmptyState>
      )}

      {docs !== null && docs.length > 0 && shown.length === 0 && <p className="py-8 text-center text-sm text-muted">No documents match “{query}”.</p>}

      {shown.length > 0 && (
        <ul className="divide-y divide-line overflow-hidden rounded-2xl border border-line bg-panel">
          {shown.map((d) => (
            <li key={d.doc_id} className="flex flex-wrap items-center gap-x-4 gap-y-2 p-4">
              <div className="grid size-10 shrink-0 place-items-center rounded-xl bg-accent-soft text-accent" aria-hidden>
                <FileText className="size-5" />
              </div>
              <div className="min-w-0 flex-1 basis-56">
                <div className="truncate text-sm font-medium" title={d.filename}>
                  {d.filename}
                </div>
                <div className="mt-0.5 text-xs text-muted">
                  {formatBytes(d.size)} · {timeAgo(d.uploaded_at)}
                  {d.uploaded_by ? ` · by ${d.uploaded_by}` : ""}
                  {d.status === "done" ? ` · ${d.chunks} passages` : ""}
                </div>
                {d.status === "failed" && d.error && <div className="mt-1 text-xs text-danger">{d.error}</div>}
              </div>
              <div className="flex flex-wrap items-center gap-1.5">
                <Badge tone="accent">level: {d.level}</Badge>
                {d.groups.length > 0 ? (
                  d.groups.map((g) => (
                    <Badge key={g}>
                      <Users className="size-3" aria-hidden /> {g}
                    </Badge>
                  ))
                ) : (
                  <Badge>all departments</Badge>
                )}
              </div>
              <StatusBadge status={d.status} />
              {canDelete(d) && (
                <button
                  onClick={() => setToDelete(d)}
                  aria-label={`Delete ${d.filename}`}
                  className="rounded-lg p-2 text-muted hover:bg-danger-soft hover:text-danger"
                >
                  <Trash2 className="size-4" />
                </button>
              )}
            </li>
          ))}
        </ul>
      )}

      <ConfirmDialog
        open={toDelete !== null}
        title="Delete this document?"
        confirmLabel="Delete"
        busy={deleting}
        onConfirm={() => void confirmDelete()}
        onCancel={() => !deleting && setToDelete(null)}
      >
        <p>
          <strong className="text-ink">{toDelete?.filename}</strong> will be removed everywhere: its searchable passages, its graph facts and its entry in this list. This cannot be undone.
        </p>
      </ConfirmDialog>
    </div>
  );
}
