import { Fragment, type ReactNode } from "react";
import type { Source } from "../api/types";
import { plainMath } from "../math";

/**
 * Renders an answer. Deliberately NOT a general markdown renderer: it understands only what the model produces here (paragraphs,
 * "- " and "1. " lists, **bold**, `code`) and turns [n] markers into citation chips. Everything is built from React elements, so
 * text from the model or from a document can never become HTML or script (React escapes it).
 */
const INLINE = /(\*\*[^*\n]+\*\*|`[^`\n]+`|\[\d+(?:\s*,\s*\d+)*\])/g;

interface Props {
  text: string;
  sources?: Source[];
  onCite?: (n: number) => void;
}

function inline(text: string, sources: Source[] | undefined, onCite: ((n: number) => void) | undefined, keyPrefix: string): ReactNode[] {
  const known = new Set((sources ?? []).map((s) => s.n));
  return text.split(INLINE).map((part, i) => {
    const key = `${keyPrefix}-${i}`;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) return <strong key={key}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2)
      return (
        <code key={key} className="rounded bg-accent-soft px-1 py-0.5 text-[0.9em]">
          {part.slice(1, -1)}
        </code>
      );
    if (/^\[\d/.test(part)) {
      const numbers = part.slice(1, -1).split(",").map((x) => Number(x.trim()));
      return (
        <Fragment key={key}>
          {numbers.map((n) =>
            known.has(n) && onCite ? (
              <button
                key={n}
                type="button"
                onClick={() => onCite(n)}
                aria-label={`Go to source ${n}`}
                className="mx-0.5 inline-flex min-w-5 items-center justify-center rounded-md bg-accent-soft px-1 text-xs font-semibold text-accent hover:bg-accent hover:text-accent-ink"
              >
                {n}
              </button>
            ) : (
              <span key={n} className="mx-0.5 text-xs text-muted" title="This number does not match any shown source">
                [{n}]
              </span>
            ),
          )}
        </Fragment>
      );
    }
    return <Fragment key={key}>{part}</Fragment>;
  });
}

export function RichText({ text, sources, onCite }: Props) {
  const blocks = plainMath(text).split(/\n{2,}/).filter((b) => b.trim() !== "");
  return (
    <div className="space-y-3 leading-relaxed">
      {blocks.map((block, bi) => {
        const lines = block.split("\n");
        const bullets = lines.every((l) => /^\s*[-*]\s+/.test(l));
        const numbered = lines.every((l) => /^\s*\d+[.)]\s+/.test(l));
        if (bullets || numbered) {
          const Tag = bullets ? "ul" : "ol";
          return (
            <Tag key={bi} className={`ml-5 space-y-1 ${bullets ? "list-disc" : "list-decimal"}`}>
              {lines.map((l, li) => (
                <li key={li}>{inline(l.replace(/^\s*(?:[-*]|\d+[.)])\s+/, ""), sources, onCite, `${bi}-${li}`)}</li>
              ))}
            </Tag>
          );
        }
        return (
          <p key={bi}>
            {lines.map((l, li) => (
              <Fragment key={li}>
                {li > 0 && <br />}
                {inline(l, sources, onCite, `${bi}-${li}`)}
              </Fragment>
            ))}
          </p>
        );
      })}
    </div>
  );
}
