import { useId } from "react";

/** A document icon that fills from top to bottom while its lines light up one after another. It is the "being read" indicator;
 *  MindTransfer sends particles from the element marked data-mind-source (this one) into the mind logo. */
export function DocFill({ className = "size-5" }: { className?: string }) {
  const clip = useId();
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden data-mind-source>
      <defs>
        <clipPath id={clip}>
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clip})`}>
        <rect className="doc-fill" x="4" y="2" width="16" height="20" fill="currentColor" stroke="none" opacity="0.35" />
      </g>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <path d="M14 2v6h6" />
      <path className="doc-line" style={{ animationDelay: "0s" }} d="M8 12h8" />
      <path className="doc-line" style={{ animationDelay: "0.35s" }} d="M8 15.5h8" />
      <path className="doc-line" style={{ animationDelay: "0.7s" }} d="M8 19h5" />
    </svg>
  );
}
