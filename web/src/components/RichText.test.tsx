import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RichText } from "./RichText";
import type { Source } from "../api/types";

const sources: Source[] = [
  { n: 1, doc_id: "a", filename: "a.pdf", chunk_index: 0, page: 2, cited: true },
  { n: 2, doc_id: "b", filename: "b.txt", chunk_index: 1, page: null, cited: false },
];

describe("RichText", () => {
  it("renders paragraphs, lists, bold and code", () => {
    render(<RichText text={"First **important** point with `code`.\n\n- one\n- two\n\n1. a\n2. b"} />);
    expect(screen.getByText("important").tagName).toBe("STRONG");
    expect(screen.getByText("code").tagName).toBe("CODE");
    expect(screen.getAllByRole("list")).toHaveLength(2);
    expect(screen.getAllByRole("listitem")).toHaveLength(4);
  });

  it("turns known citations into buttons that report their number", async () => {
    const onCite = vi.fn();
    render(<RichText text="It opened in 1889 [1] in Paris [2][1, 2]." sources={sources} onCite={onCite} />);
    expect(screen.getAllByRole("button")).toHaveLength(4); // [1] [2] [1, 2] -> 1, 2, 1, 2
    await userEvent.click(screen.getAllByRole("button", { name: "Go to source 2" })[0]);
    expect(onCite).toHaveBeenCalledWith(2);
  });

  it("does not make a button of a citation that matches no shown source", () => {
    render(<RichText text="An invented source [9]." sources={sources} onCite={() => {}} />);
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.getByText("[9]")).toBeInTheDocument();
  });

  it("never turns model or document text into HTML", () => {
    const { container } = render(<RichText text={'<img src=x onerror="alert(1)"> <script>alert(2)</script> **ok**'} sources={sources} />);
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("script")).toBeNull();
    expect(container.textContent).toContain("<script>alert(2)</script>"); // shown as plain text
  });

  it("leaves array-like text such as a[i] alone when there are no sources", () => {
    render(<RichText text="Use a[i] and [x]." />);
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("shows a calculation the model wrote in LaTeX as plain text", () => {
    const { container } = render(<RichText text={"Age:\n\n\\[ \\text{Age} = 2026 - 1981 = 45 \\]"} />);
    expect(container.textContent).toContain("Age = 2026 - 1981 = 45");
    expect(container.textContent).not.toContain("\\[");
    expect(container.textContent).not.toContain("\\text");
  });
});
