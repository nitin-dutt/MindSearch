"""The HTTP API. create_app() takes its collaborators as arguments so tests can pass fakes;
create_app_from_env() wires the real ones for uvicorn (see main.py).

Security model (alignment_plan SEC-01, SEC-04, SEC-09):
  * Every route except login needs a bearer token. The token only proves WHO the caller is; their level and
    groups are looked up from the user store on every request, so removing a user takes effect at once.
  * A caller can only upload documents at an access level they could read themselves.
  * Uploaded filenames are never used as paths; content is size- and signature-checked.
  * Logins are rate limited; logins, uploads, chats and denials are written to an audit log.
"""
import asyncio
import json
import os
import time
from typing import List, Optional

import anyio
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Path, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import auth
import conversation as conv
from access_policy import PolicyError, User, can_access
from audit import AuditLog, text_fingerprint
from doc_identity import compute_doc_id
from ratelimit import RateLimiter
from uploads import UploadError, safe_upload_path, validate_content

NO_CONTEXT_MESSAGE = (
    "I couldn't find any relevant information in the documents available to you. "
    "If you expected an answer, check that the document has finished processing and that "
    "your access level covers it."
)


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatMessage(BaseModel):
    role: str = Field(max_length=20)
    content: str = Field(max_length=conv.MAX_QUESTION_CHARS)


class ChatRequest(BaseModel):
    model: str = "rag-model"
    # The whole conversation so far, newest last. The server trims it (conversation.py); clients may send up to 60.
    messages: List[ChatMessage] = Field(min_length=1, max_length=60)
    stream: bool = True
    session_id: Optional[str] = None


def _unauthorized(detail="Not authenticated"):
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def create_app(rag, users: auth.UserStore, jwt_secret: str, upload_dir="uploads", dispatch=None,
               audit: AuditLog = None, login_limiter: RateLimiter = None, token_ttl=3600,
               retrieval_workers=None, rewriter=None) -> FastAPI:
    policy = rag.policy
    if rewriter is None:
        from llm import complete_chat
        cfg = getattr(rag, "config", None)
        rewriter = conv.QueryRewriter(getattr(cfg, "generator_model", ""), complete_chat,
                                      enabled=cfg is not None and os.getenv("REWRITE_ENABLED", "1") == "1")
    # Retrieval is CPU-heavy (embedding, cross-encoder, graph lookup). It runs in worker threads so the event
    # loop stays free for streaming and health checks, and the pool is capped so a burst of users cannot
    # oversubscribe the CPU (extra requests simply wait their turn).
    workers = int(retrieval_workers or os.getenv("RETRIEVAL_WORKERS", "4"))
    limiter_holder = []

    def retrieval_limiter():
        if not limiter_holder:                 # created lazily, inside the running event loop
            limiter_holder.append(anyio.CapacityLimiter(workers))
        return limiter_holder[0]
    for rec in users.records():          # fail at startup, not at the first login, if users.yaml is inconsistent
        policy.make_user(rec.username, rec.level, rec.groups, rec.roles)
    audit = audit or AuditLog()
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
        history = conv.trim_history(history)
        # A follow-up ("what about its price?") is rewritten into a standalone question; retrieval searches with that.
        user_query, rewritten = await rewriter.rewrite(history, question)
        t0 = time.perf_counter()
        result = await anyio.to_thread.run_sync(rag.retrieve, user_query, user, limiter=retrieval_limiter())
        context = result.to_context()
        retrieval_ms = (time.perf_counter() - t0) * 1000
        audit.log("chat", user=user.id, super_role=user.is_super, query=text_fingerprint(question),
                  rewritten=rewritten, history_messages=len(history), context_items=len(context))

        # Nothing retrieved (empty index, or nothing this user may see): answer deterministically
        # without calling the LLM, so it can't answer from thin air.
        if not context:
            async def empty_stream():
                yield f"data: {json.dumps({'choices': [{'delta': {'content': NO_CONTEXT_MESSAGE}}]})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(empty_stream(), media_type="text/event-stream",
                                     headers={"X-Retrieval-Ms": f"{retrieval_ms:.0f}"})

        # Which documents the answer is based on (filenames come from the registry; the chunk filter already
        # guaranteed this user may read them).
        sources = result.sources()
        if sources:
            names = await anyio.to_thread.run_sync(rag.registry.filenames, [x["doc_id"] for x in sources])
            sources = [{**x, "filename": names.get(x["doc_id"], "(unknown)")} for x in sources]

        async def event_stream():
            if rewritten:
                yield f"data: {json.dumps({'standalone_question': user_query})}\n\n"
            try:
                async for token in rag.stream_answer(user_query, context, conv.for_generation(history)):
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': token}}]})}\n\n"
            except Exception as e:  # noqa: BLE001
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            if sources:
                yield f"data: {json.dumps({'sources': sources})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers={"X-Retrieval-Ms": f"{retrieval_ms:.0f}"})

    # ---------------------------------------------------------------- ingest

    @app.post("/v1/ingest")
    async def ingest_endpoint(
        session_id: str = Form(...),
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
            with open(path, "wb") as out:
                out.write(raw)
            doc_id = compute_doc_id(path, acl)
            existing = await anyio.to_thread.run_sync(rag.registry.get, doc_id)
            if existing and existing.status in ("queued", "processing", "done"):
                os.remove(path)                    # same content + same access rule: nothing new to ingest
                queued.append({"filename": original_name, "doc_id": doc_id, "status": existing.status,
                               "duplicate": True, "task_id": None})
                continue
            await anyio.to_thread.run_sync(rag.registry.register, doc_id, acl, original_name, user.id, len(raw))
            try:
                task_id = dispatch(path, acl.to_payload(), doc_id) if dispatch else None
            except Exception as e:  # noqa: BLE001 - e.g. the job queue is down
                await anyio.to_thread.run_sync(rag.registry.set_status, doc_id, "failed", None, "could not queue the job")
                os.remove(path)
                audit.log("ingest_dispatch_failed", user=user.id, doc_id=doc_id, error=type(e).__name__)
                raise HTTPException(status_code=503, detail="The ingestion queue is unavailable. Try again shortly.")
            audit.log("ingest", user=user.id, super_role=user.is_super, doc_id=doc_id, filename=original_name,
                      acl=acl.to_payload(), task_id=task_id)
            queued.append({"filename": original_name, "doc_id": doc_id, "status": "queued",
                           "duplicate": False, "task_id": task_id})

        return {"status": "success", "processed": sum(1 for d in queued if not d["duplicate"]),
                "message": "Documents queued for background processing", "documents": queued}

    # ------------------------------------------------------------- documents

    DOC_ID = Path(..., pattern=r"^[0-9a-f]{32}$", description="document id")

    @app.get("/v1/documents")
    async def list_documents(user: User = Depends(current_user)):
        """Documents this user may read, newest first, with their processing status."""
        entries = await anyio.to_thread.run_sync(rag.registry.list, user)
        return {"documents": [e.to_public(policy, include_uploader=user.is_super or e.uploader == user.id) for e in entries]}

    @app.get("/v1/documents/{doc_id}")
    async def get_document(doc_id: str = DOC_ID, user: User = Depends(current_user)):
        entry = await anyio.to_thread.run_sync(rag.registry.get, doc_id, user)
        if entry is None:                          # missing and not-allowed look identical: existence is not leaked
            raise HTTPException(status_code=404, detail="Document not found")
        return entry.to_public(policy, include_uploader=user.is_super or entry.uploader == user.id)

    @app.delete("/v1/documents/{doc_id}")
    async def delete_document(doc_id: str = DOC_ID, user: User = Depends(current_user)):
        """Remove a document everywhere (chunks, graph facts, registry entry). Allowed for the uploader or a super role."""
        entry = await anyio.to_thread.run_sync(rag.registry.get, doc_id, user)
        if entry is None:
            raise HTTPException(status_code=404, detail="Document not found")
        if not (user.is_super or entry.uploader == user.id):
            audit.log("document_delete_denied", user=user.id, doc_id=doc_id)
            raise HTTPException(status_code=403, detail="Only the uploader or an administrator can delete a document.")
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
        workers = int(os.getenv("RETRIEVAL_WORKERS", "4"))
        torch.set_num_threads(int(os.getenv("TORCH_THREADS", max(1, (os.cpu_count() or 4) // workers))))
    except ImportError:
        pass


def create_app_from_env() -> FastAPI:
    from prometheus_fastapi_instrumentator import Instrumentator
    from rag_pipeline import RAGPipeline
    users = auth.UserStore.from_yaml(os.getenv("USERS_FILE", "users.yaml"))
    _set_torch_threads()
    from pipeline_config import get_preset
    # PIPELINE_PRESET picks the retrieval strategy: full (default), dense_rerank_dynamic, dense_rerank_fast, ...
    app = create_app(RAGPipeline(get_preset(os.getenv("PIPELINE_PRESET", "full"))), users, auth.load_secret(),
                     dispatch=_celery_dispatch)
    Instrumentator().instrument(app).expose(app)
    return app
