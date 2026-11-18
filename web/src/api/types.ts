/** Shapes the MindSearch API returns (see backend/api.py). */

export interface Me {
  username: string;
  level: string;
  groups: string[];
  is_super: boolean;
}

export interface Policy {
  levels: string[]; // access levels this user may put on a document they upload
  groups: string[]; // departments this user may put on a document they upload
  default_level: string;
  delete_rule: { mode: "higher_levels" | "admins_only"; text: string }; // the delete rule in force, in plain words (written by the server)
}

export type DocStatus = "queued" | "processing" | "done" | "failed";

export interface DocumentEntry {
  doc_id: string;
  filename: string;
  status: DocStatus;
  chunks: number;
  size: number;
  uploaded_at: string;
  updated_at: string;
  level: string;
  groups: string[];
  error: string | null;
  uploaded_by?: string; // only present when the API reveals it: you uploaded it, or you may delete it (a lower level's, or you are an administrator)
  can_delete: boolean; // the server's answer; the UI never works the rule out itself
  scan?: ScanReport; // only for documents that had scanned pages or pictures
}

/** What OCR (reading text out of scanned pages and pictures) could and could not read in one document. */
export interface ScanReport {
  ocr_pages: number; // scanned pages / pictures that were read and are searchable
  unreadable_pages: number[]; // PDF pages that could not be read
  unreadable_images: number; // pictures that could not be read
  skipped: number; // scanned pages not attempted (OCR unavailable, or over the per-document limit)
  junk_removed?: number; // leader-line junk ('....sscceee') removed from the OCR text; absent on documents read before 2026-11-17
}

export interface UploadResult {
  filename: string;
  doc_id: string;
  status: DocStatus;
  duplicate: boolean;
}

/** One passage the model was shown, numbered like the [n] citations in its answer. */
export interface Source {
  n: number;
  doc_id: string;
  filename: string;
  chunk_index: number | null;
  page: number | null;
  cited: boolean;
  scanned?: boolean; // the text was read from a scanned page or a picture (OCR)
}

/** One stored passage of a document, for the "view source" panel. */
export interface Passage {
  chunk_index: number;
  text: string;
  page: number | null;
  scanned?: boolean;
}

export interface PassagePage {
  doc_id: string;
  filename: string;
  total: number; // passages in the whole document
  start: number;
  passages: Passage[];
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

/** What kind of reply the server gave: a real answer, an "I don't know", or "nothing you may read matched". */
export type ReplyKind = "answered" | "refused" | "none_found";

/** What the chat stream carries. Every frame from the server names its own `type` (backend/chat.py). */
export type ChatEvent =
  | { type: "token"; text: string }
  | { type: "outcome"; outcome: ReplyKind } // stated by the server, so the UI never reads the answer text to guess it
  | { type: "standalone"; question: string }
  | { type: "searches"; queries: string[] } // the agent strategy ran several searches for one question
  | { type: "sources"; sources: Source[] }
  | { type: "error"; message: string }
  | { type: "done" };
