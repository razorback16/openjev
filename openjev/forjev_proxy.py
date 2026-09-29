"""Transparent chat forwarding; do not apply DiffusionGemma normalization."""
import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive",
       "proxy-authenticate", "proxy-authorization", "te", "trailer", "upgrade"}


def add_forjev_routes(app, settings):
    @app.get("/ready")
    async def ready(request: Request):
        try:
            response = await request.app.state.engine.client.get("/health", timeout=5)
            response.raise_for_status()
        except httpx.HTTPError:
            return JSONResponse({"status": "upstream_unavailable"}, status_code=503)
        return {"status": "ok", "backend": "forjev", "model": settings.upstream_model}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        client = httpx.AsyncClient(base_url=settings.upstream.rstrip("/"), trust_env=False,
                                   timeout=httpx.Timeout(300, connect=5))
        excluded = HOP | {"authorization", "x-origin-secret"} | {
            token.strip().lower() for token in request.headers.get("connection", "").split(",")}
        headers = {k: v for k, v in request.headers.items() if k.lower() not in excluded}
        if settings.forjev_upstream_api_key:
            headers["authorization"] = f"Bearer {settings.forjev_upstream_api_key}"
        try:
            upstream = await client.send(client.build_request(
                "POST", "/v1/chat/completions", headers=headers, content=await request.body()), stream=True)
        except httpx.HTTPError:
            await client.aclose()
            return JSONResponse({"error": {"message": "Upstream unavailable"}}, status_code=503)
        except BaseException:
            await client.aclose()
            raise

        async def close():
            await upstream.aclose()
            await client.aclose()

        excluded_response = HOP | {"content-encoding"} | {
            token.strip().lower() for token in upstream.headers.get("connection", "").split(",")}
        return StreamingResponse(upstream.aiter_bytes(), status_code=upstream.status_code,
                                 headers={k: v for k, v in upstream.headers.items()
                                          if k.lower() not in excluded_response},
                                 background=BackgroundTask(close))
