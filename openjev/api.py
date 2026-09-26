"""Jev-compatible HTTP API: POST /v1/systemone and GET /v1/models.

Request, response and error shapes follow TypeSafe's published OpenAPI 0.2.0,
so their SDKs work against this server by pointing TYPESAFE_BASE_URL at it.
Optional request fields beyond that contract (images, steps, samples, think,
sequential) are ignored by the SDKs and change nothing when left out.
POST /v1/chat/completions (openjev.chat) serves ordinary text generation.

OPENJEV_BACKEND picks what answers: DiffusionGemma through vLLM or MLX, or one
of the small encoder models (openjev.encoders). A request for a model listed in
OPENJEV_MODEL_ROUTES is passed through to the OpenJev container serving it.
"""
import base64
import binascii
import hashlib
import hmac
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, Union

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from . import __version__
from .chat import Generator, MlxGenerator, add_chat_routes
from .config import ENCODER_MODELS, Settings, served_models
from .engine import Engine, Overloaded, SchemaError, Upstream, model_ns

JSONContent = Union[str, dict[str, Any], list[Any]]
Described = Union[str, dict[str, Any], list[Any], None]


class NoulCriteria(BaseModel):
    true: Described = None
    false: Described = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Described = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Described = None
    criteria: dict[str, Described]


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Described = None
    # one level is a valid score with one possible answer, as on Jev
    criteria: list[JSONContent] = Field(min_length=1)


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class ImageObject(BaseModel):
    content_type: str
    base64: str


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str
    questions: dict[str, Question] = Field(min_length=1)
    images: list[Union[str, ImageObject]] | None = None
    steps: int | None = Field(default=None, ge=1, le=8)
    samples: int | None = Field(default=None, ge=1, le=32)
    think: int | None = Field(default=None, ge=0, le=4096)
    sequential: bool | None = None


IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def image_parts(images, settings):
    """Validate the request's images and turn them into OpenAI-style parts."""
    if len(images) > settings.max_images:
        raise SchemaError(f"at most {settings.max_images} images per request", ("body", "images"))
    parts = []
    for i, im in enumerate(images):
        loc = ("body", "images", i)
        if isinstance(im, str):
            head, sep, data = im.partition(",")
            ctype = head.removeprefix("data:").removesuffix(";base64")
            if not (sep and head.startswith("data:") and head.endswith(";base64")):
                raise SchemaError("an image is a data:image/...;base64 string or a {content_type, base64} object", loc)
        else:
            ctype, data = im.content_type, im.base64
        if ctype not in IMAGE_TYPES:
            raise SchemaError(f"image type {ctype!r} is not supported; use JPEG, PNG, WebP or GIF", loc)
        # The decoded size is 3 bytes per 4 base64 characters, minus padding. Reject
        # on that bound first: a body can carry far more base64 than the limit, and
        # decoding it to find out costs exactly the memory the limit exists to save.
        if 3 * (len(data) // 4) - 2 > settings.max_image_bytes:
            raise SchemaError(f"image data is larger than the {settings.max_image_bytes} byte limit", loc)
        try:
            size = len(base64.b64decode(data, validate=True))
        except (binascii.Error, ValueError):
            raise SchemaError("image data is not valid base64", loc) from None
        if size > settings.max_image_bytes:
            raise SchemaError(f"image is {size} bytes; the limit is {settings.max_image_bytes}", loc)
        parts.append({"type": "image_url", "image_url": {"url": f"data:{ctype};base64,{data}"}})
    return parts


def error(status, error_type, message, headers=None):
    return JSONResponse({"detail": {"error_type": error_type, "message": message}}, status_code=status, headers=headers)


log = logging.getLogger("openjev")

# A rejected body is never logged: only where it was wrong and why, so common client
# mistakes are visible without keeping anyone's data.
def log_invalid(request, parts, status=422):
    log.warning("%s %s %s", status, getattr(request.state, "request_id", "-"), "; ".join(parts) or "invalid request")


TRIM_DEPTH, TRIM_ITEMS, TRIM_CHARS = 4, 20, 500


def trim(value, depth=0):
    """A value safe to echo in an error: deep or long parts become a placeholder."""
    if isinstance(value, str):
        return value if len(value) <= TRIM_CHARS else value[:TRIM_CHARS] + "..."
    if isinstance(value, (dict, list)):
        if depth >= TRIM_DEPTH:
            return "..."
        if isinstance(value, dict):
            out = {str(k): trim(v, depth + 1) for k, v in list(value.items())[:TRIM_ITEMS]}
            if len(value) > TRIM_ITEMS:
                out["..."] = f"{len(value) - TRIM_ITEMS} more"
            return out
        return [trim(v, depth + 1) for v in value[:TRIM_ITEMS]] + (["..."] if len(value) > TRIM_ITEMS else [])
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:TRIM_CHARS]


def semantic_error(loc, msg, request=None):
    """A request whose shape is fine but whose meaning isn't, in Jev's shape: 400 with a
    plain-string detail. Field-level problems stay 422 with a list (see invalid_body)."""
    if request is not None:
        log_invalid(request, [f"{'.'.join(str(p) for p in loc)}: {msg}"], status=400)
    return JSONResponse({"detail": msg}, status_code=400)


def create_app(settings=None, tokenizer=None):
    settings = settings or Settings()
    if settings.backend not in ("vllm", "mlx", *ENCODER_MODELS):
        raise ValueError(f"unknown backend {settings.backend!r}; use one of vllm, mlx, {', '.join(ENCODER_MODELS)}")
    mlx = settings.backend == "mlx"
    encoder = settings.backend in ENCODER_MODELS
    model_version, model_names, own_models = served_models(settings.backend)
    known = {m["name"]: m for m in ENCODER_MODELS.values()}
    models_list = own_models + [known.get(name, {"name": name, "description": "", "release_date": ""})
                                for name in settings.model_routes if name not in model_names]

    @asynccontextmanager
    async def lifespan(app):
        app.state.routes = httpx.AsyncClient(timeout=httpx.Timeout(settings.forward_timeout, connect=5.0))
        if encoder:
            from .encoders import ENGINES
            app.state.engine = ENGINES[settings.backend](settings)
            yield
            await app.state.engine.close()
            await app.state.routes.aclose()
            return
        tok = tokenizer
        if tok is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(settings.mlx_model if mlx else settings.tokenizer)
        if mlx:
            from .mlx_backend import MlxEngine
        app.state.engine = (MlxEngine if mlx else Engine)(settings, tok)
        app.state.generator = (MlxGenerator(settings, app.state.engine) if mlx
                               else Generator(settings))
        yield
        await app.state.engine.close()
        await app.state.generator.close()
        await app.state.routes.aclose()

    app = FastAPI(title="OpenJev", version=__version__, lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_body(request: Request, exc: RequestValidationError):
        """Jev's 422 shape, with the reason logged. The offending value is trimmed: a
        deeply nested body used to exhaust the stack while the error was encoded."""
        if any(e.get("type") == "union_tag_invalid" for e in exc.errors()):
            # an unknown question type, which Jev answers generically
            log_invalid(request, [f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('type')}" for e in exc.errors()])
            return error(400, "api_usage_error", "Invalid request.")
        errors = []
        for e in exc.errors():
            out = {"type": e.get("type"), "loc": list(e.get("loc", ())), "msg": e.get("msg"), "input": trim(e.get("input"))}
            for extra in ("ctx", "url"):  # keep what FastAPI's own handler sends
                if extra in e:
                    out[extra] = trim(e[extra])
            errors.append(out)
        log_invalid(request, [f"{'.'.join(str(p) for p in e['loc'])}: {e['type']}" for e in errors])
        rid = getattr(request.state, "request_id", None)
        return JSONResponse({"detail": errors}, status_code=422,
                            headers={"x-typesafe-request-id": rid, "x-request-id": rid} if rid else None)

    @app.middleware("http")
    async def request_id_and_auth(request: Request, call_next):
        rid = "req_" + secrets.token_hex(16)
        request.state.request_id = rid
        if request.url.path.startswith("/v1/"):
            denied = check_auth(settings, request)
            if denied is not None:
                denied.headers["x-typesafe-request-id"] = rid
                denied.headers["x-request-id"] = rid
                return denied
            if request.method == "POST":
                # Neither uvicorn nor FastAPI bounds a body. Read it here, past auth
                # so an anonymous giant is refused before it costs any memory.
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > settings.max_body_bytes:
                        return error(413, "api_usage_error",
                                     f"request body is larger than {settings.max_body_bytes} bytes")
                request._body = bytes(body)
        spent = [0]
        model_ns.set(spent)
        started = time.perf_counter_ns()
        response = await call_next(request)
        total_ms = (time.perf_counter_ns() - started) / 1e6
        model_ms = spent[0] / 1e6
        response.headers["server-timing"] = (
            f"model;dur={model_ms:.1f}, server;dur={max(0.0, total_ms - model_ms):.1f}, total;dur={total_ms:.1f}")
        response.headers["x-typesafe-request-id"] = rid
        response.headers["x-request-id"] = rid
        return response

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"models": models_list}

    @app.post("/v1/systemone")
    async def systemone(req: SystemOneRequest, request: Request):
        if req.model in settings.model_routes and req.model not in model_names:
            return await forward(request, settings.model_routes[req.model])
        if req.model not in model_names:
            # Jev's shape for a model it doesn't serve
            return error(400, "api_usage_error", f"Unknown model: {req.model}")
        if len(req.questions) > settings.max_questions:
            # a request's questions fan out into canvas-sized groups, each a read of
            # its own; without a cap one body is unbounded work for the model
            return semantic_error(("body", "questions"),
                                  f"at most {settings.max_questions} questions per request", request)
        questions = {k: q.model_dump() for k, q in req.questions.items()}
        options = {"steps": req.steps, "samples": req.samples, "think": req.think, "sequential": req.sequential}
        engine = request.app.state.engine
        try:
            images = image_parts(req.images, settings) if req.images else None
            # Same request, same noise draws: answers are reproducible.
            key = [req.state, questions] + ([[p["image_url"]["url"] for p in images]] if images else [])
            seed = int.from_bytes(hashlib.sha256(json.dumps(key, sort_keys=True).encode()).digest()[:4], "big")
            answers, input_tokens, thought_tokens = await engine.decide(questions, req.state, seed, images, options)
        except SchemaError as e:
            return semantic_error(e.loc, str(e), request=request)
        except Upstream as e:
            return semantic_error(["body"], f"the model rejected this request: {e}", request=request)
        except Overloaded as e:
            return error(529, "overloaded_error", str(e), {"retry-after": "1"})
        except httpx.HTTPError as e:
            return error(503, "api_error", f"inference backend unavailable: {type(e).__name__}", {"retry-after": "2"})
        # output_tokens stays 0 as in Jev's contract unless a thought was generated
        return {"model": model_version, "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": thought_tokens}}

    if not encoder:
        add_chat_routes(app)

    return app


FORWARD_HEADERS = ("authorization", "x-origin-secret", "content-type")


async def forward(request, url):
    """Pass a request through to the container serving its model, and its answer back
    unchanged: that container applies the same contract and its own auth."""
    headers = {h: request.headers[h] for h in FORWARD_HEADERS if h in request.headers}
    started = time.perf_counter_ns()
    try:
        r = await request.app.state.routes.post(url + "/v1/systemone", content=await request.body(), headers=headers)
    except httpx.HTTPError as e:
        return error(503, "api_error", f"inference backend unavailable: {type(e).__name__}", {"retry-after": "2"})
    finally:
        # the other container's time, network included, is the model's time here
        spent = model_ns.get()
        if spent is not None:
            spent[0] += time.perf_counter_ns() - started
    keep = {h: r.headers[h] for h in ("content-type", "retry-after") if h in r.headers}
    return Response(r.content, status_code=r.status_code, headers=keep)


def check_auth(settings, request):
    # compare as bytes: compare_digest on str raises TypeError for non-ASCII,
    # and a header is latin-1 decoded, so a non-ASCII credential was a 500
    if settings.origin_secret:
        got = request.headers.get("x-origin-secret", "")
        if not hmac.compare_digest(got.encode(), settings.origin_secret.encode()):
            return error(403, "permission_error", "Direct access to this origin is not allowed.")
    if settings.api_key:
        auth = request.headers.get("authorization", "")
        if not auth:
            return error(403, "authentication_error", "Must supply an API key! Check your request and try again.")
        token = auth.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(token.encode(), settings.api_key.encode()):
            return error(401, "authentication_error", "Cannot authenticate with the server. Please check your API key and try again.")
    return None
