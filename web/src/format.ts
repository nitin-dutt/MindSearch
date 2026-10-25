export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

const RELATIVE = new Intl.RelativeTimeFormat("en", { numeric: "auto" });

/** "2 minutes ago" for a recent ISO timestamp, a plain date for an old one. */
export function timeAgo(iso: string, now: Date = new Date()): string {
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return "";
  const seconds = Math.round((then.getTime() - now.getTime()) / 1000);
  const abs = Math.abs(seconds);
  if (abs < 45) return "just now";
  if (abs < 3600) return RELATIVE.format(Math.round(seconds / 60), "minute");
  if (abs < 86400) return RELATIVE.format(Math.round(seconds / 3600), "hour");
  if (abs < 86400 * 14) return RELATIVE.format(Math.round(seconds / 86400), "day");
  return then.toLocaleDateString("en", { year: "numeric", month: "short", day: "numeric" });
}

export const ABSTAIN_SENTENCE = "I don't know based on the documents available to you.";

/** The three special things the system can say instead of an answer, recognised so the UI can show them differently. */
export function answerKind(text: string): "answer" | "refusal" | "none_found" {
  const norm = text.replace(/’/g, "'").trim().toLowerCase();
  if (norm.includes(ABSTAIN_SENTENCE.toLowerCase()) || norm.startsWith("i don't know") || norm.startsWith("i do not know")) return "refusal";
  if (norm.startsWith("i couldn't find any relevant information")) return "none_found";
  return "answer";
}
