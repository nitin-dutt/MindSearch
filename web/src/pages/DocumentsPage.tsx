import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { AlertCircle, CheckCircle2, Clock, Eye, FileText, Loader2, Search, Trash2, Upload, Users } from "lucide-react";
import { deleteDocument, listDocuments } from "../api/client";
import type { DocStatus, DocumentEntry, ScanReport } from "../api/types";
import { PassageViewer, type ViewTarget } from "../components/PassageViewer";
import { Badge, ConfirmDialog, EmptyState, Page, PageHeader, Skeleton, buttonClass } from "../components/ui";
import { useToast } from "../components/Toast";
import { formatBytes, timeAgo } from "../format";

export function StatusBadge({ status }: { status: DocStatus }) {
  if (status === "done")
    return (
      <Badge tone="ok">
        <CheckCircle2 className="size-3.5" aria-hidden /> Ready
      </Badge>
    );
  if (status === "failed")
    return (
      <Badge tone="danger">
        <AlertCircle className="size-3.5" aria-hidden /> Failed
      </Badge>
    );
  if (status === "processing")
    return (
      <Badge tone="warn">
        <Loader2 className="size-3.5 animate-spin" aria-hidden /> Processing
      </Badge>
    );
  return (
    <Badge tone="neutral">
      <Clock className="size-3.5" aria-hidden /> Queued
    </Badge>
  );
}

/** What OCR did with a document's scanned pages, in one line: what is searchable, and what the user should know is missing. */
export function scanNote(scan: ScanReport): string {
  const parts: string[] = [];
  if (scan.ocr_pages > 0) parts.push(`${scan.ocr_pages} scanned page${scan.ocr_pages === 1 ? "" : "s"} read`);
  if (scan.unreadable_pages.length > 0) parts.push(`could not read page${scan.unreadable_pages.length === 1 ? "" : "s"} ${scan.unreadable_pages.join(", ")}`);
  if (scan.unreadable_images > 0) parts.push(`could not read ${scan.unreadable_images} picture${scan.unreadable_images === 1 ? "" : "s"}`);
  if (scan.skipped > 0) parts.push(`${scan.skipped} scanned page${scan.skipped === 1 ? "" : "s"} not read`);
  return parts.join(" · ");
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
  const [viewing, setViewing] = useState<ViewTarget | null>(null);

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

  // The server decides who may delete (an administrator, or a higher level than the uploader; never the uploader). The UI only shows its answer.
  const canDelete = (d: DocumentEntry) => d.can_delete;

  return (
    <Page>
      <PageHeader
        title="Documents"
        description="Everything your account is allowed to read, with who else can read it."
        action={
          <Link to="/upload" className={buttonClass("primary")}>
            <Upload className="size-4" aria-hidden /> Upload
          </Link>
        }
      />

      {docs !== null && docs.length > 0 && (
        <label className="relative block">
          <span className="sr-only">Filter documents</span>
          <Search className="pointer-events-none absolute left-3 top-3 size-4 text-muted" aria-hidden />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter by name, level or department…"
            className="h-10 w-full rounded-lg border border-line bg-panel pl-10 pr-3 text-sm"
          />
        </label>
      )}

      {error && (
        <div role="alert" className="rounded-lg border border-danger/30 bg-danger-soft px-4 py-3 text-sm text-danger">
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
          Upload a PDF, Word, text, Markdown or image file and choose who may read it. It becomes searchable once processing finishes.
        </EmptyState>
      )}

      {docs !== null && docs.length > 0 && shown.length === 0 && <p className="py-8 text-center text-sm text-muted">No documents match “{query}”.</p>}

      {shown.length > 0 && (
        <ul className="divide-y divide-line overflow-hidden rounded-2xl border border-line bg-panel">
          {shown.map((d) => (
            // Status and actions sit in fixed-width slots at the end of every row, so they form straight columns down the list
            // even when a row has no "view" button or carries extra lines (an error, a scan note).
            <li key={d.doc_id} className="flex flex-wrap items-center gap-x-4 gap-y-2 p-4">
              <div className="grid size-10 shrink-0 place-items-center rounded-lg bg-accent-soft text-accent" aria-hidden>
                <FileText className="size-5" />
              </div>
              <div className="min-w-0 flex-1 basis-56">
                <div className="truncate text-sm font-medium" title={d.filename}>
                  {d.filename}
                </div>
                <div className="mt-1 text-xs text-muted">
                  {formatBytes(d.size)} · {timeAgo(d.uploaded_at)}
                  {d.uploaded_by ? ` · by ${d.uploaded_by}` : ""}
                  {d.status === "done" ? ` · ${d.chunks} passages` : ""}
                </div>
                {d.status === "failed" && d.error && <div className="mt-1 text-xs text-danger">{d.error}</div>}
                {d.status === "done" && d.scan && (
                  <div className={`mt-1 text-xs ${d.scan.unreadable_pages.length + d.scan.unreadable_images + d.scan.skipped > 0 ? "text-warn" : "text-muted"}`}>{scanNote(d.scan)}</div>
                )}
              </div>
              <div className="flex flex-wrap items-center gap-1 md:justify-end">
                <Badge tone="accent">level: {d.level}</Badge>
                {d.groups.length > 0 ? (
                  d.groups.map((g) => (
                    <Badge key={g}>
                      <Users className="size-3.5" aria-hidden /> {g}
                    </Badge>
                  ))
                ) : (
                  <Badge>all departments</Badge>
                )}
              </div>
              <div className="w-28 shrink-0">
                <StatusBadge status={d.status} />
              </div>
              <div className="flex w-17 shrink-0 justify-end gap-1">
              {d.status === "done" ? (
                <button
                  onClick={() => setViewing({ doc_id: d.doc_id, filename: d.filename, chunk_index: null })}
                  aria-label={`View ${d.filename}`}
                  title="Read the indexed text"
                  className="grid size-8 place-items-center rounded-lg text-muted transition-colors hover:bg-accent-soft hover:text-ink"
                >
                  <Eye className="size-4" />
                </button>
              ) : (
                <span className="size-8" aria-hidden />
              )}
              {canDelete(d) ? (
                <button
                  onClick={() => setToDelete(d)}
                  aria-label={`Delete ${d.filename}`}
                  className="grid size-8 place-items-center rounded-lg text-muted transition-colors hover:bg-danger-soft hover:text-danger"
                >
                  <Trash2 className="size-4" />
                </button>
              ) : (
                <span className="size-8" aria-hidden />
              )}
              </div>
            </li>
          ))}
        </ul>
      )}

      <PassageViewer target={viewing} onClose={() => setViewing(null)} />

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
    </Page>
  );
}
