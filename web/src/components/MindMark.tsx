// A brain split down the middle: the left half is the natural brain (an outline with a few folds), the right half is the same brain as a
// circuit (traces that end in nodes). It says "a mind, built from your documents, run by software" in one picture.
const LEFT_OUTLINE =
  "M16 4 C13.6 2.6 10.4 3.2 9.6 5.6 C7 5.4 5.6 7.6 6.4 9.4 C4.2 10.6 3.8 13.2 5 14.8 C3.8 16.6 4.6 19.2 6.8 19.8 C7 22.2 9.6 23.6 11.8 22.6 C12.8 24.2 14.8 24.6 16 24";
// one short notch inward from each place where two lobes meet, and two long folds flowing out from the middle line
const FOLDS = [
  "M9.6 5.6 C10.6 6.6 10.8 7.6 10.6 8.6",
  "M6.4 9.4 C7.6 9.6 8.6 10.2 9 11",
  "M5 14.8 C6.4 14.4 7.6 14.4 8.6 14.8",
  "M6.8 19.8 C7.8 19.2 8.6 18.6 9 17.6",
  "M11.8 22.6 C12.2 21.6 12.2 20.6 12 19.8",
  "M16 9.4 C13.4 9.2 11.6 10.4 11 12.4",
  "M16 15.2 C13.6 15 12 16 11.4 17.6",
];
// traces from the middle line to the right; each ends in a node
const TRACES = [
  "M16 7.2 H18.6 L20.6 9.2 H22.4",
  "M18.6 7.2 L20 5.8 H21.2",
  "M16 10.8 H19.6 L21.2 12.4 H24.6",
  "M16 14.2 H20.2 L22 16 H25.6",
  "M16 17.6 H18.6 L20.4 19.4 H23.4",
  "M16 21 H18 L19.6 22.6 H21.2",
  "M21.2 12.4 L22.6 11 H24.2",
];
const NODES: [number, number][] = [
  [23.5, 9.2],
  [22.3, 5.8],
  [25.8, 12.4],
  [25.4, 11],
  [26.8, 16],
  [24.6, 19.4],
  [22.3, 22.6],
];

/** The MindSearch symbol on its own, drawn in one solid colour (the accent), with no button, tile, gradient or glow behind it. */
export function MindMark({ className = "size-8" }: { className?: string }) {
  const paint = "var(--accent)";
  return (
    <span className={`relative inline-grid shrink-0 place-items-center ${className}`} aria-hidden>
      <svg
        viewBox="0 0 32 28"
        className="size-full overflow-visible"
        fill="none"
        stroke={paint}
        strokeWidth="1.25"
        strokeLinecap="round"
        strokeLinejoin="round"
        focusable="false"
      >
        <path d={LEFT_OUTLINE} />
        {FOLDS.map((d) => (
          <path key={d} d={d} />
        ))}
        <path d="M16 4 V24.4" strokeWidth="1.5" />
        {TRACES.map((d) => (
          <path key={d} d={d} strokeWidth="1" />
        ))}
        {NODES.map(([x, y]) => (
          <circle key={`${x}-${y}`} cx={x} cy={y} r="1" fill={paint} stroke="none" />
        ))}
      </svg>
    </span>
  );
}
