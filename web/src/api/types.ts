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
  uploaded_by?: string; // only present when the API reveals it: you may delete the document (your own, a lower level's, or you are an administrator)
  can_delete: boolean; // the server's answer; the UI never works the rule out itself
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
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

/** What the chat stream can carry besides the answer text. */
export type ChatEvent =
  | { type: "token"; text: string }
  | { type: "standalone"; question: string }
  | { type: "sources"; sources: Source[] }
  | { type: "error"; message: string }
  | { type: "done" };
