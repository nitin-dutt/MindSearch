"""The HTTP API. create_app() takes its collaborators as arguments so tests can pass fakes;
create_app_from_env() wires the real ones for uvicorn (see main.py).

Security model (alignment_plan SEC-01, SEC-04, SEC-09):
  * Every route except login needs a bearer token. The token only proves WHO the caller is; their level and
    groups are looked up from the user store on every request, so removing a user takes effect at once.
  * A caller can only upload documents at an access level they could read themselves.
  * Uploaded filenames are never used as paths; content is size- and signature-checked.
  * Logins are rate limited; logins, uploads, chats and denials are written to an audit log.
"""
import json
import os
from typing import List, Optional

import anyio
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Path, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import auth
import chat
import conversation as conv
from answers import ABSTAIN_MESSAGE  # noqa: F401 - part of this module's public names (tests and tools read it from here)
from chat import NO_CONTEXT_MESSAGE  # noqa: F401
from llm import LLMError
from settings import get_settings
from access_policy import PolicyError, User, can_access
from audit import AuditLog
from ingestion import Ingestion, QueueUnavailable
from ratelimit import RateLimiter
from uploads import UploadError, safe_upload_path, validate_content


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatMessage(BaseModel):
    role: str = Field(max_length=20)
    content: str = Field(max_length=conv.MAX_QUESTION_CHARS)


class ChatRequest(BaseModel):
    # The whole conversation so far, newest last. The server trims it (conversation.py); clients may send up to 60.
    messages: List[ChatMessage] = Field(min_length=1, max_length=60)
    # Limit the search to these documents (the user picked a file). Access rules still apply on top: this can only narrow.
    doc_ids: Optional[List[str]] = Field(default=None, max_length=20)


def _unauthorized(detail="Not authenticated"):
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def _sse(event: dict) -> str:
    """One chat event (chat.py) as a Server-Sent Event. Every frame carries its `type`; a token frame also keeps the OpenAI-style
    `choices` shape, so tools written for that format (the load test) read the answer text unchanged."""
    if event["type"] == "token":
        event = {"type": "token", "choices": [{"delta": {"content": event["text"]}}]}
    return f"data: {json.dumps(event)}\n\n"


def create_app(rag, users: auth.UserStore, jwt_secret: str, upload_dir="uploads", dispatch=None,
               audit: AuditLog = None, login_limiter: RateLimiter = None, token_ttl=3600,
               retrieval_workers=None, rewriter=None, planner=None) -> FastAPI:
    """`rewriter` / `planner` are the chat helpers that need the language model (chat.model_helpers); without them every question is
    searched once, as typed."""
    policy = rag.policy
    audit = audit or AuditLog()
    chats = chat.ChatService(rag, audit, planner=planner, rewriter=rewriter, workers=retrieval_workers)
    ingest = Ingestion(rag, dispatch)
    for rec in users.records():          # fail at startup, not at the first login, if users.yaml is inconsistent
        policy.make_user(rec.username, rec.level, rec.groups, rec.roles)
    limiter = login_limiter or RateLimiter()
    os.makedirs(upload_dir, exist_ok=True)

    app = FastAPI()

    def current_user(authorization: Optional[str] = Header(default=None)) -> User:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise _unauthorized()
        try:
            claims = auth.verify_token(authorization[7:].strip(), jwt_secret)
        except auth.AuthError:
            audit.log("auth_invalid_token")
            raise _unauthorized("Invalid or expired token")
        record = users.get(claims.get("sub"))
        if record is None:
            audit.log("auth_unknown_subject", sub=str(claims.get("sub")))
            raise _unauthorized("Invalid or expired token")
        return policy.make_user(record.username, record.level, record.groups, record.roles)

    @app.get("/healthz")
    def healthz():
        """Liveness probe. Deliberately trivial: under load it answers instantly only if the event loop is free."""
        return {"status": "ok"}

    # ------------------------------------------------------------------ auth

    @app.post("/v1/auth/login")
    def login(req: LoginRequest, request: Request):
        key = f"{request.client.host if request.client else 'unknown'}|{req.username.casefold()}"
        if not limiter.allow(key):
            audit.log("login_rate_limited", username=req.username)
            raise HTTPException(status_code=429, detail="Too many attempts. Try again in a minute.",
                                headers={"Retry-After": "60"})
        record = users.authenticate(req.username, req.password)
        if record is None:
            audit.log("login_failure", username=req.username)
            raise _unauthorized("Incorrect username or password")
        limiter.reset(key)
        audit.log("login_success", username=record.username)
        return {"access_token": auth.create_token({"sub": record.username}, jwt_secret, token_ttl),
                "token_type": "bearer", "expires_in": token_ttl}

    @app.get("/v1/me")
    def me(user: User = Depends(current_user)):
        return {"username": user.id, "level": policy.level_name(user.rank), "groups": list(user.groups),
                "is_super": user.is_super}

    @app.get("/v1/policy")
    def assignable(user: User = Depends(current_user)):
        """Levels and groups this user may put on a document they upload."""
        levels = policy.level_names if user.is_super else [n for n in policy.level_names if policy.rank_of(n) <= user.rank]
        groups = list(policy.groups) if user.is_super else list(user.groups)
        return {"levels": levels, "groups": groups, "default_level": policy.level_name(user.rank)}

    # ------------------------------------------------------------------ chat

    @app.post("/v1/chat/completions")
    async def chat_endpoint(req: ChatRequest, user: User = Depends(current_user)):
        try:
            question, history = conv.split_conversation(req.messages)
        except conv.ConversationError as e:
            raise HTTPException(status_code=422, detail=str(e))
        # Everything about answering lives in chat.py. A model that cannot even start is a real HTTP failure, not a 200 stream
        # whose "answer" is an error message.
        try:
            turn = await chats.begin(history, question, user, req.doc_ids)
        except LLMError:
            raise HTTPException(status_code=503, detail="The answer service is unavailable. Please try again in a moment.")

        async def event_stream():
            async for event in turn.events():
                yield _sse(event)
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers={"X-Retrieval-Ms": f"{turn.retrieval_ms:.0f}"})

    # ---------------------------------------------------------------- ingest

    @app.post("/v1/ingest")
    async def ingest_endpoint(
        files: List[UploadFile] = File(...),
        level: Optional[str] = Form(None),
        groups: str = Form(""),
        user: User = Depends(current_user),
    ):
        """Queue documents. `level` / `groups` say who may read them; they default to the uploader's own
        level and groups, and an uploader cannot grant access wider than their own."""
        wanted_groups = [g.strip() for g in groups.split(",") if g.strip()] or list(user.groups)
        try:
            acl = policy.make_acl(level or policy.level_name(user.rank), wanted_groups)
        except PolicyError as e:
            raise HTTPException(status_code=422, detail=str(e))
        if not can_access(user, acl):
            audit.log("ingest_denied", user=user.id, level=level, groups=wanted_groups)
            raise HTTPException(status_code=403, detail="You can only upload documents at an access level you could read yourself.")

        prepared = []
        for f in files:                       # validate everything before saving anything
            raw = await f.read()
            try:
                path = safe_upload_path(upload_dir, f.filename)
                validate_content(path, raw)
            except UploadError as e:
                raise HTTPException(status_code=e.status_code, detail=f"{f.filename}: {e}")
            prepared.append((f.filename, path, raw))

        queued = []
        for original_name, path, raw in prepared:
            try:
                receipt = await anyio.to_thread.run_sync(ingest.submit, path, raw, acl, original_name, user.id)
            except QueueUnavailable as e:
                audit.log("ingest_dispatch_failed", user=user.id, doc_id=e.doc_id, error=e.cause)
                raise HTTPException(status_code=503, detail="The ingestion queue is unavailable. Try again shortly.")
            if not receipt["duplicate"]:
                audit.log("ingest", user=user.id, super_role=user.is_super, doc_id=receipt["doc_id"], filename=original_name,
                          acl=acl.to_payload(), task_id=receipt["task_id"])
            queued.append(receipt)

        return {"status": "success", "processed": sum(1 for d in queued if not d["duplicate"]),
                "message": "Documents queued for background processing", "documents": queued}

    # ------------------------------------------------------------- documents

    DOC_ID = Path(..., pattern=r"^[0-9a-f]{32}$", description="document id")

    def may_delete(user: User, entry) -> bool:
        """Who can delete a document: a super role, the uploader, or someone of a HIGHER level than the uploader (a manager can remove
        what an employee uploaded, never the other way round, and equals cannot remove each other's). The uploader's level is looked up
        in the user store; an uploader who no longer exists, or who is a super role, can only be overruled by a super role."""
        if user.is_super or entry.uploader == user.id:
            return True
        uploader = users.get(entry.uploader)
        if uploader is None:
            return False
        if policy.make_user(uploader.username, uploader.level, uploader.groups, uploader.roles).is_super:
            return False                          # an administrator's upload can only be removed by an administrator, whatever their level
        return user.rank > policy.rank_of(uploader.level)

    def public(entry, user: User) -> dict:
        deletable = may_delete(user, entry)
        out = entry.to_public(policy, include_uploader=deletable)       # people who may delete a document may see who uploaded it
        out["can_delete"] = deletable
        return out

    @app.get("/v1/documents")
    async def list_documents(user: User = Depends(current_user)):
        """Documents this user may read, newest first, with their processing status."""
        entries = await anyio.to_thread.run_sync(rag.registry.list, user)
        return {"documents": [public(e, user) for e in entries]}

    @app.get("/v1/documents/{doc_id}")
    async def get_document(doc_id: str = DOC_ID, user: User = Depends(current_user)):
        entry = await anyio.to_thread.run_sync(rag.registry.get, doc_id, user)
        if entry is None:                          # missing and not-allowed look identical: existence is not leaked
            raise HTTPException(status_code=404, detail="Document not found")
        return public(entry, user)

    @app.delete("/v1/documents/{doc_id}")
    async def delete_document(doc_id: str = DOC_ID, user: User = Depends(current_user)):
        """Remove a document everywhere (chunks, graph facts, registry entry). Allowed for the uploader, a super role, or a higher level than the uploader."""
        entry = await anyio.to_thread.run_sync(rag.registry.get, doc_id, user)
        if entry is None:
            raise HTTPException(status_code=404, detail="Document not found")
        if not may_delete(user, entry):
            audit.log("document_delete_denied", user=user.id, doc_id=doc_id)
            raise HTTPException(status_code=403, detail="Only the uploader, an administrator or someone of a higher level than the uploader can delete a document.")
        report = await anyio.to_thread.run_sync(rag.delete_document, doc_id)
        audit.log("document_delete", user=user.id, super_role=user.is_super, doc_id=doc_id)
        return {"status": "deleted", "doc_id": doc_id, **report}

    return app


def _celery_dispatch(path, acl_payload, doc_id):
    from worker import process_document_task
    return process_document_task.delay(path, acl_payload, doc_id).id


def _set_torch_threads():
    """Several retrievals now run at once; give each a share of the CPU instead of every one grabbing all cores."""
    try:
        import torch
        cfg = get_settings()
        torch.set_num_threads(cfg.torch_threads or max(1, (os.cpu_count() or 4) // cfg.retrieval_workers))
    except ImportError:
        pass


def create_app_from_env() -> FastAPI:
    from prometheus_fastapi_instrumentator import Instrumentator
    from rag_pipeline import RAGPipeline
    cfg = get_settings()
    users = auth.UserStore.from_yaml(cfg.users_file)
    _set_torch_threads()
    from pipeline_config import get_preset
    # PIPELINE_PRESET picks the retrieval strategy: hybrid_graph_chunks (default), dense_rerank_dynamic, dense_rerank_fast, ...
    app = create_app(RAGPipeline(get_preset(cfg.pipeline_preset)), users, auth.load_secret(),
                     dispatch=_celery_dispatch)
    Instrumentator().instrument(app).expose(app)
    return app
