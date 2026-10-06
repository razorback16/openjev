"""Numeric route installed INSIDE the existing vLLM HTTP process.

The initial engine bridge avoids OpenAI chat serialization. It deliberately
does not claim to bypass the sampler: that needs a runner-specific provider.
vLLM imports are lazy so CPU tests and the ForJev adapter need no GPU packages.
"""
import asyncio
from collections.abc import Mapping
import uuid
from typing import Annotated

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from .decision_scores import parse_response, score_response


TokenID = Annotated[StrictInt, Field(ge=0)]


class DecisionScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    messages: list[dict] = Field(min_length=1)
    candidate_token_ids: list[TokenID] = Field(min_length=2, max_length=255)
    chat_template_kwargs: dict = Field(default_factory=lambda: {"enable_thinking": False})
    require_prefill: bool = False
    skip_reading_prefix_cache: StrictBool = False


async def render_prompt(chat, request):
    """Use vLLM's own multimodal chat rendering, including its server template."""
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    except ImportError:
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    if request.chat_template_kwargs != {"enable_thinking": False}:
        raise HTTPException(400, "Decision scoring requires enable_thinking=false only")
    chat_request = ChatCompletionRequest(
        model=request.model, messages=request.messages,
        chat_template_kwargs=request.chat_template_kwargs,
        max_tokens=1, temperature=0, stream=False,
    )
    error = await chat._check_model(chat_request)
    if error is not None:
        raise HTTPException(400, "The requested model is not available")
    # Do not accidentally score the base weights for a requested LoRA alias.
    adapter = chat._maybe_get_adapters(chat_request)
    if adapter is not None:
        raise HTTPException(400, "Decision scoring currently supports the resident base model only")
    if hasattr(chat, "render_chat_request"):
        rendered = await chat.render_chat_request(chat_request)
        if not isinstance(rendered, tuple) or len(rendered) != 2:
            raise HTTPException(400, "vLLM could not render the scoring prompt")
        _, prompts = rendered
    elif hasattr(chat, "_preprocess_chat"):
        tokenizer = await chat.engine_client.get_tokenizer()
        _, _, prompts = await chat._preprocess_chat(
            chat_request, tokenizer, chat_request.messages,
            chat_template=chat.chat_template,
            chat_template_content_format=chat.chat_template_content_format,
            add_generation_prompt=True, continue_final_message=False,
            tool_dicts=None, documents=None,
            chat_template_kwargs=chat_request.chat_template_kwargs,
            tool_parser=None, add_special_tokens=chat_request.add_special_tokens,
        )
    else:
        raise HTTPException(501, "Unsupported vLLM renderer; use a checked runtime integration")
    if len(prompts) != 1:
        raise HTTPException(400, "Decision scoring requires one rendered prompt")
    return prompts[0]


async def engine_scores(chat, prompt, token_ids, *, request_id, disconnected,
                        skip_reading_prefix_cache=False):
    """Read raw candidate logprobs directly from AsyncLLM, never chat formatting.

One internal generation token is still scheduled. Missing rows remain errors;
this route cannot repair an upstream sampler/transport defect.
"""
    from vllm.sampling_params import SamplingParams
    if getattr(chat.model_config, "logprobs_mode", "raw_logprobs") != "raw_logprobs":
        raise HTTPException(501, "engine_scores requires vLLM logprobs_mode=raw_logprobs")
    max_logprobs = getattr(chat.model_config, "max_logprobs", None)
    if max_logprobs is not None and len(token_ids) > max_logprobs:
        raise HTTPException(400, "Candidate count exceeds vLLM max_logprobs")
    params = SamplingParams(max_tokens=1, temperature=0, seed=0, n=1,
                            logprobs=len(token_ids), logprob_token_ids=token_ids,
                            detokenize=False, ignore_eos=True,
                            skip_reading_prefix_cache=skip_reading_prefix_cache)
    engine = chat.engine_client
    generator = engine.generate(prompt, params, request_id)
    result = None
    completed = False
    try:
        async for output in generator:
            if await disconnected():
                raise asyncio.CancelledError()
            result = output
        completed = True
    finally:
        if not completed:
            await engine.abort(request_id)
        await generator.aclose()
    if result is None or len(result.outputs) != 1:
        raise HTTPException(502, "No complete scoring result from vLLM")
    output = result.outputs[0]
    if (output.finish_reason == "error" or len(output.token_ids) != 1
            or output.logprobs is None or len(output.logprobs) != 1):
        raise HTTPException(502, "vLLM returned misaligned candidate logprobs")
    row = output.logprobs[0]
    if not isinstance(row, Mapping):
        raise HTTPException(502, "vLLM returned an empty candidate logprob row")
    missing = [t for t in token_ids if t not in row]
    if missing:
        raise HTTPException(502, {"message": "vLLM omitted requested candidate logprobs",
                                  "missing_token_ids": missing})
    try:
        values = [row[t].logprob for t in token_ids]
        return score_response(token_ids, values, len(result.prompt_token_ids),
                              score_type="raw_logprobs", execution="engine_logprobs")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(502, "vLLM returned incomplete or invalid candidate logprobs") from exc


def install_routes(app, *, render=render_prompt, score=engine_scores, prefill_score=None):
    """Attach to vLLM's app and resident engine; preserve its /v1 authentication.

`prefill_score` is a runner-specific async provider with the same signature as
`engine_scores`. It must return a verified prefill_logits result; supplying no
provider makes require_prefill fail before rendering or inference.
"""
    if any(getattr(route, "path", None) == "/v1/decision_scores" for route in app.routes):
        raise RuntimeError("Decision scoring route is already installed")

    @app.post("/v1/decision_scores")
    async def decision_scores(body: DecisionScoreRequest, request: Request):
        if len(set(body.candidate_token_ids)) != len(body.candidate_token_ids):
            raise HTTPException(400, "Candidate token IDs must be unique")
        if body.require_prefill and prefill_score is None:
            raise HTTPException(501, "Prefill-only runner integration is not installed")
        chat = getattr(request.app.state, "openai_serving_chat", None)
        if chat is None:
            raise HTTPException(503, "vLLM chat renderer is not ready")
        vocab_size = chat.model_config.get_vocab_size()
        if any(t >= vocab_size for t in body.candidate_token_ids):
            raise HTTPException(400, "Candidate token ID is outside the model vocabulary")
        prompt = await render(chat, body)
        provider = prefill_score if body.require_prefill else score
        cache_options = ({"skip_reading_prefix_cache": True}
                         if body.skip_reading_prefix_cache else {})
        data = await provider(chat, prompt, body.candidate_token_ids,
                              request_id="forjev-score-" + uuid.uuid4().hex,
                              disconnected=request.is_disconnected, **cache_options)
        try:
            parse_response(data, body.candidate_token_ids, require_prefill=body.require_prefill)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise HTTPException(502, "Invalid candidate score provider result") from exc
        return JSONResponse(data)
