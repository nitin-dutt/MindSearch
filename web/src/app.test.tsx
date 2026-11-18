import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ME, event, json, mockApi, renderApp, sse, tokenEvent } from "./test-utils";
import type { DocumentEntry, Source } from "./api/types";

afterEach(() => vi.unstubAllGlobals());

const POLICY = { levels: ["Staff", "Manager"], groups: ["HR", "Finance"], default_level: "Manager", delete_rule: { mode: "higher_levels" as const, text: "A document can be deleted by an administrator, or by someone of a higher level than the person who uploaded it. The uploader cannot delete their own document." } };
const doc = (over: Partial<DocumentEntry>): DocumentEntry => ({
  doc_id: "d".repeat(32), filename: "handbook.pdf", status: "done", chunks: 12, size: 2048, uploaded_at: "2026-10-17T10:00:00Z",
  updated_at: "2026-10-17T10:01:00Z", level: "Staff", groups: [], error: null, can_delete: false, ...over,
});
const SOURCES: Source[] = [
  { n: 1, doc_id: "a", filename: "handbook.pdf", chunk_index: 4, page: 3, cited: true },
  { n: 2, doc_id: "b", filename: "memo.txt", chunk_index: 0, page: null, cited: false },
];

describe("sign in", () => {
  it("shows the login screen when there is no session", () => {
    renderApp("/", { signedIn: false });
    expect(screen.getByRole("heading", { name: "Sign in" })).toBeInTheDocument();
  });

  it("explains a wrong password and a rate limit in plain words", async () => {
    mockApi({
      "POST /v1/auth/login": ({ body }) => (JSON.parse(String(body)).password === "wrong" ? json({ detail: "Incorrect username or password" }, 401) : json({ detail: "Too many attempts." }, 429)),
    });
    renderApp("/", { signedIn: false });
    await userEvent.type(screen.getByLabelText("Username"), "alice");
    await userEvent.type(screen.getByLabelText("Password"), "wrong");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Incorrect username or password.");
    await userEvent.clear(screen.getByLabelText("Password"));
    await userEvent.type(screen.getByLabelText("Password"), "again");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByText(/Too many attempts/)).toBeInTheDocument();
  });

  it("signs in, then shows who you are and your level", async () => {
    mockApi({ "POST /v1/auth/login": () => ({ access_token: "t" }), "GET /v1/me": () => ME });
    renderApp("/", { signedIn: false });
    await userEvent.type(screen.getByLabelText("Username"), "alice");
    await userEvent.type(screen.getByLabelText("Password"), "pw");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByText("Ask about your documents")).toBeInTheDocument();
    expect(screen.getAllByText("alice").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Manager").length).toBeGreaterThan(0);
  });

  it("falls back to the login screen when a stored session has expired", async () => {
    mockApi({ "GET /v1/me": () => json({ detail: "Invalid or expired token" }, 401) });
    renderApp("/");
    expect(await screen.findByRole("heading", { name: "Sign in" })).toBeInTheDocument();
  });
});

describe("chat", () => {
  async function openChat(chatHandler: Parameters<typeof mockApi>[0][string]) {
    const api = mockApi({ "GET /v1/me": () => ME, "POST /v1/chat/completions": chatHandler });
    renderApp("/");
    await screen.findByText("Ask about your documents");
    return api;
  }
  const ask = async (text: string) => {
    await userEvent.type(screen.getByLabelText("Your question"), text);
    await userEvent.keyboard("{Enter}");
  };

  it("streams an answer with clickable citations and a sources list that marks the cited ones", async () => {
    await openChat(() => sse([event({ type: "standalone", question: "What is the leave policy?" }), tokenEvent("Staff get 25 days "), tokenEvent("of leave [1]."), event({ type: "sources", sources: SOURCES }), "data: [DONE]\n\n"]));
    await ask("and leave?");
    expect(await screen.findByText(/Staff get 25 days of leave/)).toBeInTheDocument();
    expect(screen.getByText(/Searched for: What is the leave policy\?/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Sources used \(1\)/ })).toHaveAttribute("aria-expanded", "false"); // hidden until asked for
    expect(screen.queryByText("handbook.pdf")).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: "Go to source 1" })); // clicking a citation opens the list
    expect(screen.getByRole("button", { name: /Sources used \(1\)/ })).toHaveAttribute("aria-expanded", "true");
    const used = screen.getAllByRole("list")[0];
    expect(within(used).getByText("handbook.pdf")).toBeInTheDocument();
    expect(within(used).getByText(/page 3/)).toBeInTheDocument();
    expect(within(used).queryByText("memo.txt")).toBeNull(); // the passage that was not used is tucked away
    expect(screen.getByText("1 more passage looked at, not used")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Sources used \(1\)/ })); // the header closes it again
    expect(screen.queryByText("handbook.pdf")).toBeNull();
  });

  it("opens the cited passage in context when a source is clicked, and says when text was read from a scan", async () => {
    const scanned: Source[] = [{ n: 1, doc_id: "a".repeat(32), filename: "lease-scan.pdf", chunk_index: 4, page: 3, cited: true, scanned: true }];
    const api = mockApi({
      "GET /v1/me": () => ME,
      "POST /v1/chat/completions": () => sse([tokenEvent("The lease runs nine years [1]."), event({ type: "sources", sources: scanned }), "data: [DONE]\n\n"]),
      // the passage before the cited one comes first, so the citation is read in context
      [`GET /v1/documents/${"a".repeat(32)}/passages?start=3&limit=3`]: () => ({
        doc_id: "a".repeat(32), filename: "lease-scan.pdf", total: 9, start: 3,
        passages: [
          { chunk_index: 3, text: "The parties met in Rotterdam.", page: 2 },
          { chunk_index: 4, text: "The lease runs for nine years.", page: 3, scanned: true },
          { chunk_index: 5, text: "Rent is reviewed every year.", page: 3 },
        ],
      }),
      [`GET /v1/documents/${"a".repeat(32)}/passages?start=6&limit=3`]: () => ({ doc_id: "a".repeat(32), filename: "lease-scan.pdf", total: 9, start: 6, passages: [{ chunk_index: 6, text: "Either party may end it.", page: 4 }] }),
    });
    renderApp("/");
    await screen.findByText("Ask about your documents");
    await ask("how long is the lease?");
    await userEvent.click(await screen.findByRole("button", { name: /Sources used \(1\)/ }));
    expect(screen.getByText(/page 3 \(scanned\)/)).toBeInTheDocument(); // the citation says the text came from a scan
    await userEvent.click(screen.getByRole("button", { name: "View source 1: lease-scan.pdf", hidden: true }));
    expect(await screen.findByText("The lease runs for nine years.")).toBeInTheDocument();
    expect(screen.getByText("The parties met in Rotterdam.")).toBeInTheDocument();
    expect(screen.getByText("cited passage")).toBeInTheDocument();
    expect(screen.getByText("Passage 5 of 9")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Later", hidden: true }));
    expect(await screen.findByText("Either party may end it.")).toBeInTheDocument();
    expect(api.calls.filter((c) => c.key.includes("/passages")).length).toBe(2);
  });

  it("explains it when the source document can no longer be read", async () => {
    mockApi({
      "GET /v1/me": () => ME,
      "POST /v1/chat/completions": () => sse([tokenEvent("Staff get 25 days [1]."), event({ type: "sources", sources: SOURCES }), "data: [DONE]\n\n"]),
      "GET /v1/documents/a/passages?start=3&limit=3": () => json({ detail: "Document not found" }, 404),
    });
    renderApp("/");
    await screen.findByText("Ask about your documents");
    await ask("leave?");
    await userEvent.click(await screen.findByRole("button", { name: /Sources used \(1\)/ }));
    await userEvent.click(screen.getByRole("button", { name: "View source 1: handbook.pdf", hidden: true }));
    expect(await screen.findByText("This document is no longer available to you.")).toBeInTheDocument();
  });

  it("shows the separate searches when one question covered several things", async () => {
    await openChat(() => sse([event({ type: "searches", queries: ["refund policy", "warranty terms"] }), tokenEvent("Refunds take 14 days [1]."), event({ type: "sources", sources: SOURCES }), "data: [DONE]\n\n"]));
    await ask("compare the refund policy with the warranty terms");
    expect(await screen.findByText(/Refunds take 14 days/)).toBeInTheDocument();
    const searches = screen.getByLabelText("Searches run for this question");
    expect(within(searches).getByText("refund policy")).toBeInTheDocument();
    expect(within(searches).getByText("warranty terms")).toBeInTheDocument();
  });

  it("searches everything by default and only the chosen file after picking one", async () => {
    const docs = [doc({ doc_id: "a".repeat(32), filename: "handbook.pdf" }), doc({ doc_id: "b".repeat(32), filename: "memo.txt" })];
    const api = mockApi({
      "GET /v1/me": () => ME,
      "GET /v1/documents": () => ({ documents: docs }),
      "POST /v1/chat/completions": () => sse([tokenEvent("Fine."), "data: [DONE]\n\n"]),
    });
    renderApp("/");
    const picker = await screen.findByRole("combobox", { name: "Search in" });
    await ask("anything");
    await screen.findByText("Fine.");
    await userEvent.selectOptions(picker, "memo.txt");
    await ask("again");
    await waitFor(() => expect(api.calls.filter((c) => c.key.includes("chat")).length).toBe(2));
    const [first, second] = api.calls.filter((c) => c.key.includes("chat")).map((c) => JSON.parse(String(c.body)));
    expect(first.doc_ids).toBeUndefined();
    expect(second.doc_ids).toEqual(["b".repeat(32)]);
    expect(await screen.findByText("in memo.txt")).toBeInTheDocument(); // the thread shows which file a question was limited to
  });

  it("shows a reply the server calls a refusal as a refusal, not as a normal answer", async () => {
    await openChat(() => sse([event({ type: "outcome", outcome: "refused" }), tokenEvent("I don't know based on the documents available to you."), "data: [DONE]\n\n"]));
    await ask("what is the airspeed of a swallow?");
    expect(await screen.findByText(/nothing was guessed/)).toBeInTheDocument();
  });

  it("shows a nothing-found reply with the hint about processing and access", async () => {
    await openChat(() => sse([event({ type: "outcome", outcome: "none_found" }), tokenEvent("I couldn't find any relevant information."), "data: [DONE]\n\n"]));
    await ask("anything about llamas?");
    expect(await screen.findByText(/check that the document finished processing/)).toBeInTheDocument();
  });

  it("goes by what the server says, not by how the text reads", async () => {
    await openChat(() => sse([tokenEvent("I don't know why the report says so, but the price is 5."), event({ type: "outcome", outcome: "answered" }), "data: [DONE]\n\n"]));
    await ask("what is the price?");
    expect(await screen.findByText(/the price is 5/)).toBeInTheDocument();
    expect(screen.queryByText(/nothing was guessed/)).not.toBeInTheDocument();
  });

  it("shows a model outage as a warning with the server's message and no answer bubble text", async () => {
    await openChat(() => json({ detail: "The answer service is unavailable. Please try again in a moment." }, 503));
    await ask("hello?");
    expect(await screen.findByRole("alert")).toHaveTextContent("The answer service is unavailable");
  });

  it("keeps the partial answer and adds a warning when the answer is interrupted", async () => {
    await openChat(() => sse([tokenEvent("The permits renew in"), event({ type: "error", message: "The answer was interrupted. Please try again." }), "data: [DONE]\n\n"]));
    await ask("when do permits renew?");
    expect(await screen.findByText(/The permits renew in/)).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("interrupted");
  });

  it("sends earlier turns with a follow-up but leaves failed attempts out of the history", async () => {
    let call = 0;
    const { calls } = await openChat(() => {
      call += 1;
      return call === 1 ? json({ detail: "The answer service is unavailable." }, 503) : sse([tokenEvent("Fine."), "data: [DONE]\n\n"]);
    });
    await ask("first question");
    await screen.findByRole("alert");
    await ask("second question");
    await screen.findByText("Fine.");
    const body = JSON.parse(String(calls.filter((c) => c.key === "POST /v1/chat/completions")[1].body));
    expect(body.messages.map((m: { content: string }) => m.content)).toEqual(["first question", "second question"]);
    expect(body.messages.every((m: { role: string }) => m.role === "user")).toBe(true); // the failed answer was not sent back
  });

  it("starts a fresh conversation on request", async () => {
    await openChat(() => sse([tokenEvent("Hi there."), "data: [DONE]\n\n"]));
    await ask("hello");
    await screen.findByText("Hi there.");
    await userEvent.click(screen.getByRole("button", { name: /New chat/ }));
    expect(await screen.findByText("Ask about your documents")).toBeInTheDocument();
  });
});

describe("documents", () => {
  const mine = doc({ doc_id: "a".repeat(32), filename: "mine.pdf", uploaded_by: "alice", level: "Manager", groups: ["HR"], can_delete: true });
  const theirs = doc({ doc_id: "b".repeat(32), filename: "theirs.docx" }); // the API hides the uploader of other people's documents

  it("lists documents with level, departments and status, and lets you delete only your own", async () => {
    mockApi({ "GET /v1/me": () => ME, "GET /v1/documents": () => ({ documents: [mine, theirs, doc({ doc_id: "c".repeat(32), filename: "busy.txt", status: "processing" })] }) });
    renderApp("/documents");
    expect(await screen.findByText("mine.pdf")).toBeInTheDocument();
    expect(screen.getByText("level: Manager")).toBeInTheDocument();
    expect(screen.getByText("HR")).toBeInTheDocument();
    expect(screen.getAllByText("all departments").length).toBeGreaterThan(0);
    expect(screen.getByText("Processing")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Delete mine.pdf" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete theirs.docx" })).toBeNull();
  });

  it("shows a delete button wherever the server says the user may delete (an administrator, or a higher level than the uploader)", async () => {
    const lower = doc({ doc_id: "d".repeat(32), filename: "from-staff.txt", uploaded_by: "bob", can_delete: true });
    mockApi({ "GET /v1/me": () => ME, "GET /v1/documents": () => ({ documents: [lower, theirs] }) });
    renderApp("/documents");
    expect(await screen.findByRole("button", { name: "Delete from-staff.txt" })).toBeInTheDocument();
    expect(screen.getByText(/by bob/)).toBeInTheDocument(); // the person who may delete it is told who uploaded it
    expect(screen.queryByRole("button", { name: "Delete theirs.docx" })).toBeNull();
  });

  it("asks for confirmation, deletes, and refreshes the list", async () => {
    let docs = [mine];
    const { calls } = mockApi({
      "GET /v1/me": () => ME,
      "GET /v1/documents": () => ({ documents: docs }),
      [`DELETE /v1/documents/${mine.doc_id}`]: () => {
        docs = [];
        return { status: "deleted" };
      },
    });
    renderApp("/documents");
    await userEvent.click(await screen.findByRole("button", { name: "Delete mine.pdf" }));
    expect(await screen.findByText(/cannot be undone/)).toBeInTheDocument();
    expect(calls.some((c) => c.key.startsWith("DELETE"))).toBe(false); // nothing deleted before confirming
    await userEvent.click(screen.getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(calls.some((c) => c.key.startsWith("DELETE"))).toBe(true));
    expect(await screen.findByText("No documents yet")).toBeInTheDocument();
  });

  it("shows a friendly empty state and filters the list", async () => {
    mockApi({ "GET /v1/me": () => ME, "GET /v1/documents": () => ({ documents: [mine, theirs] }) });
    renderApp("/documents");
    await screen.findByText("mine.pdf");
    await userEvent.type(screen.getByPlaceholderText(/Filter by name/), "theirs");
    expect(screen.queryByText("mine.pdf")).toBeNull();
    expect(screen.getByText("theirs.docx")).toBeInTheDocument();
    await userEvent.clear(screen.getByPlaceholderText(/Filter by name/));
    await userEvent.type(screen.getByPlaceholderText(/Filter by name/), "zzz");
    expect(screen.getByText(/No documents match/)).toBeInTheDocument();
  });

  it("says which scanned pages were read and which could not be, and opens a document's text from its start", async () => {
    const scan = doc({ doc_id: "e".repeat(32), filename: "contract-scan.pdf", scan: { ocr_pages: 4, unreadable_pages: [2, 7], unreadable_images: 0, skipped: 0 } });
    mockApi({
      "GET /v1/me": () => ME,
      "GET /v1/documents": () => ({ documents: [scan, doc({ doc_id: "f".repeat(32), filename: "queued.txt", status: "queued" })] }),
      [`GET /v1/documents/${"e".repeat(32)}/passages?start=0&limit=3`]: () => ({ doc_id: "e".repeat(32), filename: "contract-scan.pdf", total: 1, start: 0, passages: [{ chunk_index: 0, text: "This agreement is made in Lisbon.", page: 1, scanned: true }] }),
    });
    renderApp("/documents");
    expect(await screen.findByText("4 scanned pages read · could not read pages 2, 7")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "View queued.txt" })).toBeNull(); // nothing to read until processing has finished
    await userEvent.click(screen.getByRole("button", { name: "View contract-scan.pdf" }));
    expect(await screen.findByText("This agreement is made in Lisbon.")).toBeInTheDocument();
    expect(screen.queryByText("cited passage")).toBeNull(); // opened from the list, not from a citation
  });

  it("explains a failed document", async () => {
    mockApi({ "GET /v1/me": () => ME, "GET /v1/documents": () => ({ documents: [doc({ status: "failed", error: "RuntimeError: ingestion failed after retries" })] }) });
    renderApp("/documents");
    expect(await screen.findByText("Failed")).toBeInTheDocument();
    expect(screen.getByText(/ingestion failed after retries/)).toBeInTheDocument();
  });
});

describe("upload", () => {
  it("offers only the levels and departments the policy allows and sends the choice with the file", async () => {
    const { calls } = mockApi({
      "GET /v1/me": () => ME,
      "GET /v1/policy": () => POLICY,
      "GET /v1/documents": () => ({ documents: [] }),
      "POST /v1/ingest": () => ({ documents: [{ filename: "note.txt", doc_id: "e".repeat(32), status: "queued", duplicate: false }] }),
    });
    renderApp("/upload");
    const level = (await screen.findByLabelText("Access level")) as HTMLSelectElement;
    await waitFor(() => expect(level.value).toBe("Manager"));
    expect(within(level).getAllByRole("option").map((o) => o.textContent)).toEqual(["Staff", "Manager"]);
    await userEvent.selectOptions(level, "Staff");
    await userEvent.click(screen.getByLabelText("HR"));
    await userEvent.upload(screen.getByLabelText("Choose files"), new File(["hello"], "note.txt", { type: "text/plain" }));
    expect(screen.getByText("note.txt")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Upload 1 file/ }));
    expect(await screen.findByText("Just uploaded")).toBeInTheDocument();
    const form = calls.find((c) => c.key === "POST /v1/ingest")!.body as FormData;
    expect(form.get("level")).toBe("Staff");
    expect(form.get("groups")).toBe("HR");
  });

  it("tells the user when a file was already uploaded", async () => {
    mockApi({
      "GET /v1/me": () => ME,
      "GET /v1/policy": () => POLICY,
      "GET /v1/documents": () => ({ documents: [doc({ doc_id: "f".repeat(32), filename: "dup.txt" })] }),
      "POST /v1/ingest": () => ({ documents: [{ filename: "dup.txt", doc_id: "f".repeat(32), status: "done", duplicate: true }] }),
    });
    renderApp("/upload");
    await screen.findByLabelText("Access level");
    await userEvent.upload(screen.getByLabelText("Choose files"), new File(["x"], "dup.txt", { type: "text/plain" }));
    await userEvent.click(screen.getByRole("button", { name: /Upload 1 file/ }));
    expect(await screen.findByText("already uploaded (same content and access)")).toBeInTheDocument(); // on the file
    expect(screen.getByText(/Nothing new/)).toBeInTheDocument(); // and the summary message
  });

  it("does not send a file that is already in the library, and says why", async () => {
    const { calls } = mockApi({
      "GET /v1/me": () => ME,
      "GET /v1/policy": () => POLICY,
      "GET /v1/documents": () => ({ documents: [doc({ filename: "report.pdf", size: 5, level: "Staff" })] }),
      "POST /v1/ingest": () => ({ documents: [{ filename: "new.txt", doc_id: "9".repeat(32), status: "queued", duplicate: false }] }),
    });
    renderApp("/upload");
    await screen.findByLabelText("Access level");
    await userEvent.upload(screen.getByLabelText("Choose files"), new File(["12345"], "report.pdf", { type: "application/pdf" })); // same name and size
    expect(await screen.findByText("already uploaded (Staff)")).toBeInTheDocument();
    expect(screen.getByText(/will be skipped/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Upload" })).toBeDisabled(); // nothing new to send
    await userEvent.upload(screen.getByLabelText("Choose files"), new File(["hello"], "new.txt", { type: "text/plain" }));
    await userEvent.click(screen.getByRole("button", { name: /Upload 1 file/ })); // only the new file is counted and sent
    await screen.findByText("Just uploaded");
    const sent = (calls.find((c) => c.key === "POST /v1/ingest")!.body as FormData).getAll("files") as File[];
    expect(sent.map((f) => f.name)).toEqual(["new.txt"]);
  });

  it("accepts pictures, which the server reads with OCR", async () => {
    mockApi({ "GET /v1/me": () => ME, "GET /v1/policy": () => POLICY, "GET /v1/documents": () => ({ documents: [] }) });
    renderApp("/upload");
    await screen.findByLabelText("Access level");
    await userEvent.upload(screen.getByLabelText("Choose files") as HTMLInputElement, new File(["x"], "whiteboard.PNG"), { applyAccept: false });
    expect(await screen.findByText("whiteboard.PNG")).toBeInTheDocument();
    expect(screen.queryByText(/skipped/)).toBeNull();
  });

  it("skips unsupported file types with a message", async () => {
    mockApi({ "GET /v1/me": () => ME, "GET /v1/policy": () => POLICY, "GET /v1/documents": () => ({ documents: [] }) });
    renderApp("/upload");
    await screen.findByLabelText("Access level");
    const input = screen.getByLabelText("Choose files") as HTMLInputElement;
    await userEvent.upload(input, new File(["x"], "virus.exe"), { applyAccept: false });
    expect(await screen.findByText(/skipped/)).toBeInTheDocument();
    expect(screen.queryByText("virus.exe")).toBeNull();
  });
});

describe("my access", () => {
  it("explains what a normal user can read and share", async () => {
    mockApi({ "GET /v1/me": () => ME, "GET /v1/policy": () => POLICY });
    renderApp("/access");
    expect(await screen.findByText(/Documents at level/)).toBeInTheDocument();
    expect(screen.getByText(/never searched, quoted or counted/)).toBeInTheDocument();
    expect(await screen.findByText("Staff")).toBeInTheDocument();
    // the delete rule is the server's wording, shown as it is: the page does not work the rule out itself
    expect(screen.getByText(/The uploader cannot delete their own document/)).toBeInTheDocument();
  });

  it("says so plainly for an administrator", async () => {
    mockApi({ "GET /v1/me": () => ({ ...ME, is_super: true }), "GET /v1/policy": () => POLICY });
    renderApp("/access");
    expect(await screen.findByText(/you can read every document/)).toBeInTheDocument();
    expect(screen.getAllByText(/written to the audit log/).length).toBeGreaterThan(0);
    expect(screen.getByText(/As an administrator you can delete any document/)).toBeInTheDocument();
  });
});

describe("sidebar", () => {
  it("collapses to icons, keeps every link reachable by its name, and remembers the choice", async () => {
    mockApi({ "GET /v1/me": () => ME });
    renderApp("/");
    await userEvent.click(await screen.findByRole("button", { name: "Collapse sidebar" }));
    expect(screen.getByRole("button", { name: "Expand sidebar" })).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("link", { name: "Documents" })).toBeInTheDocument(); // the label is hidden visually, not for screen readers
    expect(screen.getByRole("button", { name: "Sign out" })).toBeInTheDocument();
    expect(localStorage.getItem("mindsearch.sidebar")).toBe("collapsed");
    await userEvent.click(screen.getByRole("button", { name: "Expand sidebar" }));
    expect(localStorage.getItem("mindsearch.sidebar")).toBe("open");
  });
});
