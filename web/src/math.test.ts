import { plainMath } from "./math";

describe("plainMath", () => {
  it("turns the calculation that was shown as raw LaTeX into readable text", () => {
    expect(plainMath("So:\n\\[ \\text{Age} = 2026 - 1981 = 45 \\]\nHe is 45.")).toBe("So:\n\nAge = 2026 - 1981 = 45\n\nHe is 45.");
  });

  it("handles inline maths, symbols and fractions", () => {
    expect(plainMath("It costs \\( 3 \\times 4 \\approx 12 \\) in total")).toBe("It costs 3 × 4 ≈ 12 in total");
    expect(plainMath("\\[ x = \\frac{10}{4} \\]")).toBe("\n\nx = (10) / (4)\n\n");
    expect(plainMath("$$ a \\leq b \\geq c \\neq d \\pm 1 $$")).toContain("a ≤ b ≥ c ≠ d ± 1");
  });

  it("leaves ordinary text, prices and unknown commands alone", () => {
    expect(plainMath("The fee is $5 and the other is $10.")).toBe("The fee is $5 and the other is $10.");
    expect(plainMath("Nothing special here [1].")).toBe("Nothing special here [1].");
    expect(plainMath("\\( \\alpha + 1 \\)")).toBe("\\alpha + 1");
  });
});
