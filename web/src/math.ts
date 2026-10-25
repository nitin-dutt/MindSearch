// Models sometimes write a calculation in LaTeX ("\[ \text{Age} = 2026 - 1981 = 45 \]"). The page does not render LaTeX (it would need a large maths
// library for something rare here), so this turns the common cases into plain readable text. Anything it does not recognise is left as written.

const SYMBOLS: [RegExp, string][] = [
  [/\\times\b/g, "×"],
  [/\\cdot\b/g, "·"],
  [/\\div\b/g, "÷"],
  [/\\approx\b/g, "≈"],
  [/\\(?:leq?|leqslant)\b/g, "≤"],
  [/\\(?:geq?|geqslant)\b/g, "≥"],
  [/\\neq?\b/g, "≠"],
  [/\\pm\b/g, "±"],
  [/\\%/g, "%"],
  [/\\[,;:!]|\\ /g, " "],
];

/** One piece of LaTeX maths as plain text. */
function simplify(latex: string): string {
  let s = latex;
  s = s.replace(/\\(?:text|mathrm|mathbf|textbf|operatorname)\s*\{([^{}]*)\}/g, "$1");
  s = s.replace(/\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, "($1) / ($2)");
  for (const [pattern, replacement] of SYMBOLS) s = s.replace(pattern, replacement);
  return s.replace(/\s+/g, " ").trim();
}

/** Replaces \[ ... \], $$ ... $$ (each on its own line) and \( ... \) with plain text. Single dollar signs are never touched (they are prices). */
export function plainMath(text: string): string {
  return text
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, m: string) => `\n\n${simplify(m)}\n\n`)
    .replace(/\$\$([\s\S]*?)\$\$/g, (_, m: string) => `\n\n${simplify(m)}\n\n`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, m: string) => simplify(m))
    .replace(/\n{3,}/g, "\n\n");
}
