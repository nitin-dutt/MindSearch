import { useEffect, useRef, useState, type DragEvent } from "react";
import { Link } from "react-router-dom";
import { CloudUpload, FileText, X } from "lucide-react";
import { getPolicy, uploadFiles } from "../api/client";
import type { Policy, UploadResult } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { Badge, Button, Card, Skeleton } from "../components/ui";
import { useToast } from "../components/Toast";
import { DocFill } from "../components/DocFill";
import { MindTransfer } from "../components/MindTransfer";
import { formatBytes } from "../format";
import { StatusBadge, useDocuments } from "./DocumentsPage";

const ACCEPT = ".pdf,.docx,.txt,.md";
const MIN_EFFECT_MS = 3400; // a small file uploads in a blink; keep the "document goes into the mind" animation on screen long enough to see

export function UploadPage() {
  const { user } = useAuth();
  const toast = useToast();
  const { docs, refresh } = useDocuments();
  const [policy, setPolicy] = useState<Policy | null>(null);
  const [level, setLevel] = useState("");
  const [groups, setGroups] = useState<string[]>([]);
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [reading, setReading] = useState(false); // the animation runs while uploading, and for at least MIN_EFFECT_MS
  const [results, setResults] = useState<UploadResult[] | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getPolicy()
      .then((p) => {
        setPolicy(p);
        setLevel(p.default_level);
      })
      .catch((e) => toast.error(e instanceof Error ? e.message : "Could not load your upload options."));
  }, [toast]);

  function addFiles(list: FileList | File[]) {
    const incoming = Array.from(list).filter((f) => ACCEPT.split(",").some((ext) => f.name.toLowerCase().endsWith(ext)));
    const rejected = Array.from(list).length - incoming.length;
    if (rejected > 0) toast.error(`${rejected} file${rejected > 1 ? "s were" : " was"} skipped: only PDF, Word, text and Markdown files are supported.`);
    setFiles((current) => [...current, ...incoming.filter((f) => !current.some((c) => c.name === f.name && c.size === f.size))]);
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    addFiles(e.dataTransfer.files);
  }

  // The list only holds documents this user may read, so matching on name and size reveals nothing. A file that is already there is not sent again.
  const existing = (f: File) => docs?.find((d) => d.filename === f.name && d.size === f.size);
  const fresh = files.filter((f) => !existing(f));
  const alreadyThere = files.length - fresh.length;

  async function submit() {
    if (fresh.length === 0 || !level) return;
    setReading(true);
    const started = Date.now();
    try {
      const out = await uploadFiles(fresh, level, groups);
      setResults(out);
      setFiles([]);
      const queued = out.filter((r) => !r.duplicate).length;
      toast.success(queued > 0 ? `${queued} document${queued > 1 ? "s" : ""} queued for processing.` : "Nothing new: these documents were already uploaded.");
      await refresh();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      window.setTimeout(() => setReading(false), Math.max(0, MIN_EFFECT_MS - (Date.now() - started)));
    }
  }

  const toggleGroup = (g: string) => setGroups((cur) => (cur.includes(g) ? cur.filter((x) => x !== g) : [...cur, g]));
  const statusOf = (docId: string) => docs?.find((d) => d.doc_id === docId)?.status;

  return (
    <div className="mx-auto max-w-3xl space-y-5 p-4 md:p-8">
      <div>
        <h1 className="text-lg font-semibold">Upload documents</h1>
        <p className="text-sm text-muted">PDF, Word, text or Markdown. Choose who may read them before you upload.</p>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`rounded-2xl border-2 border-dashed p-8 text-center transition ${dragging ? "border-accent bg-accent-soft" : "border-line bg-panel"}`}
      >
        <CloudUpload className="mx-auto size-8 text-accent" aria-hidden />
        <p className="mt-3 text-sm">
          Drag files here, or{" "}
          <button type="button" onClick={() => input.current?.click()} className="font-semibold text-accent underline underline-offset-2">
            browse
          </button>
        </p>
        <input ref={input} type="file" multiple accept={ACCEPT} className="sr-only" aria-label="Choose files" onChange={(e) => e.target.files && addFiles(e.target.files)} />
      </div>

      {files.length > 0 && (
        <ul className="divide-y divide-line rounded-2xl border border-line bg-panel">
          {files.map((f) => (
            <li key={f.name + f.size} className="flex items-center gap-3 p-3 text-sm">
              <FileText className="size-4 shrink-0 text-muted" aria-hidden />
              <span className={`min-w-0 flex-1 truncate ${existing(f) ? "text-muted" : ""}`}>{f.name}</span>
              {existing(f) && <Badge tone="warn">already uploaded ({existing(f)?.level})</Badge>}
              <span className="text-xs text-muted">{formatBytes(f.size)}</span>
              <button aria-label={`Remove ${f.name}`} onClick={() => setFiles((c) => c.filter((x) => x !== f))} className="text-muted hover:text-danger">
                <X className="size-4" />
              </button>
            </li>
          ))}
        </ul>
      )}

      {alreadyThere > 0 && (
        <p role="status" className="rounded-xl border border-warn/40 bg-warn-soft px-3.5 py-2.5 text-sm text-ink">
          {alreadyThere === 1 ? "This file is" : `${alreadyThere} files are`} already in your library and will be skipped. To change who can read {alreadyThere === 1 ? "it" : "them"}, delete {alreadyThere === 1 ? "it" : "them"} from Documents and upload again.
        </p>
      )}

      <Card className="space-y-4">
        <h2 className="text-sm font-semibold">Who can read these documents?</h2>
        {!policy ? (
          <Skeleton className="h-20" />
        ) : (
          <>
            <label className="block text-sm font-medium">
              Access level
              <select value={level} onChange={(e) => setLevel(e.target.value)} className="mt-1 block w-full rounded-lg border border-line bg-bg px-3 py-2 text-sm sm:w-64">
                {policy.levels.map((l) => (
                  <option key={l}>{l}</option>
                ))}
              </select>
            </label>
            {policy.groups.length > 0 && (
              <fieldset>
                <legend className="text-sm font-medium">Departments</legend>
                <div className="mt-2 flex flex-wrap gap-2">
                  {policy.groups.map((g) => (
                    <label key={g} className={`flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-1.5 text-sm ${groups.includes(g) ? "border-accent bg-accent-soft" : "border-line"}`}>
                      <input type="checkbox" checked={groups.includes(g)} onChange={() => toggleGroup(g)} className="accent-[var(--accent)]" />
                      {g}
                    </label>
                  ))}
                </div>
              </fieldset>
            )}
            <p className="rounded-lg bg-bg p-3 text-sm text-muted">
              People at level <strong className="text-ink">{level}</strong> or higher can read these
              {groups.length > 0 ? (
                <>
                  , if they are in: <strong className="text-ink">{groups.join(", ")}</strong>.
                </>
              ) : user && user.groups.length > 0 ? (
                <>
                  , if they share one of <em>your</em> departments (<strong className="text-ink">{user.groups.join(", ")}</strong>), because no department was chosen.
                </>
              ) : (
                <>, in any department.</>
              )}{" "}
              You can only grant access up to your own level.
            </p>
          </>
        )}
        <Button onClick={() => void submit()} disabled={fresh.length === 0 || !level || reading} className="w-full sm:w-auto" aria-busy={reading}>
          {reading ? <DocFill className="size-5" /> : null}
          {reading ? "Sending to the mind…" : `Upload ${fresh.length > 0 ? `${fresh.length} file${fresh.length > 1 ? "s" : ""}` : ""}`}
        </Button>
        <MindTransfer active={reading} />
      </Card>

      {results && (
        <Card className="space-y-3">
          <h2 className="text-sm font-semibold">Just uploaded</h2>
          <ul className="space-y-2">
            {results.map((r) => (
              <li key={r.doc_id} className="flex flex-wrap items-center gap-2 text-sm">
                <FileText className="size-4 text-muted" aria-hidden />
                <span className="min-w-0 flex-1 truncate">{r.filename}</span>
                {r.duplicate && <Badge>already uploaded (same content and access)</Badge>}
                <StatusBadge status={statusOf(r.doc_id) ?? r.status} />
              </li>
            ))}
          </ul>
          <Link to="/documents" className="text-sm font-medium text-accent hover:underline">
            View all documents →
          </Link>
        </Card>
      )}
    </div>
  );
}
