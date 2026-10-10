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
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import auth
from access_policy import PolicyError, User, can_access
from audit import AuditLog, text_fingerprint
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
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "rag-model"
    messages: List[ChatMessage]
    stream: bool = True
    session_id: Optional[str] = None


def _unauthorized(detail="Not authenticated"):
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def create_app(rag, users: auth.UserStore, jwt_secret: str, upload_dir="uploads", dispatch=None,
               audit: AuditLog = None, login_limiter: RateLimiter = None, token_ttl=3600) -> FastAPI:
    policy = rag.policy
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
        user_query = req.messages[-1].content
        context = rag.retrieve_context(user_query, user)
        audit.log("chat", user=user.id, super_role=user.is_super, query=text_fingerprint(user_query),
                  context_items=len(context))

        # Nothing retrieved (empty index, or nothing this user may see): answer deterministically
        # without calling the LLM, so it can't answer from thin air.
        if not context:
            async def empty_stream():
                yield f"data: {json.dumps({'choices': [{'delta': {'content': NO_CONTEXT_MESSAGE}}]})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(empty_stream(), media_type="text/event-stream")

        async def event_stream():
            try:
                async for token in rag.stream_answer(user_query, context):
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': token}}]})}\n\n"
                    await asyncio.sleep(0.01)
            except Exception as e:  # noqa: BLE001
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

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
            task_id = dispatch(path, acl.to_payload()) if dispatch else None
            audit.log("ingest", user=user.id, super_role=user.is_super, filename=original_name,
                      stored_as=os.path.basename(path), acl=acl.to_payload(), task_id=task_id)
            queued.append({"filename": original_name, "task_id": task_id})

        return {"status": "success", "processed": len(queued),
                "message": "Documents queued for background processing", "documents": queued}

    return app


def _celery_dispatch(path, acl_payload):
    from worker import process_document_task
    return process_document_task.delay(path, acl_payload).id


def create_app_from_env() -> FastAPI:
    from prometheus_fastapi_instrumentator import Instrumentator
    from rag_pipeline import RAGPipeline
    users = auth.UserStore.from_yaml(os.getenv("USERS_FILE", "users.yaml"))
    app = create_app(RAGPipeline(), users, auth.load_secret(), dispatch=_celery_dispatch)
    Instrumentator().instrument(app).expose(app)
    return app
