"""Prefill readout hooks for the fingerprinted B12X V2 runtime.

No model is constructed here. Numeric rows use the engine's pooling transport;
the request remains a generation request for rendering and admission purposes.
"""
import asyncio

from fastapi import HTTPException

from .decision_scores import score_response


MARKER = "forjev_prefill_v1"
VERSION = "0.1.dev20759+gb40673cd0.d20260913"


def candidates(params):
    extra = getattr(params, "extra_args", None) or {}
    ids = extra.get(MARKER)
    if ids is None:
        return None
    if (not isinstance(ids, list) or not 2 <= len(ids) <= 255
            or any(type(t) is not int or t < 0 for t in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Invalid native decision candidate IDs")
    return ids


def is_score(request):
    return candidates(request.sampling_params) is not None


def check_config(config):
    if not config.use_v2_model_runner:
        raise ValueError("Native decision scoring requires the B12X V2 model runner")
    parallel = config.parallel_config
    for name in ("tensor_parallel_size", "pipeline_parallel_size",
                 "data_parallel_size", "decode_context_parallel_size",
                 "prefill_context_parallel_size"):
        if getattr(parallel, name, 1) != 1:
            raise ValueError("Native decision scoring currently requires TP/PP/DP/CP=1")
    model = config.model_config
    if (getattr(model, "is_encoder_decoder", False)
            or getattr(model, "is_diffusion", False)
            or getattr(config, "is_mm_encoder_only", False)
            or getattr(model, "is_mm_encoder_only", False)
            or getattr(model, "runner_type", "generate") != "generate"):
        raise ValueError("Native decision scoring requires an autoregressive decoder generation runner")
    if getattr(config, "kv_transfer_config", None) is not None:
        raise ValueError("Native decision scoring does not support remote KV transfer")


def validate_request(scheduler, request):
    if not is_score(request):
        return
    check_config(scheduler.vllm_config)
    params = request.sampling_params
    if (params.max_tokens != 1 or params.n != 1 or params.logprobs is not None
            or params.prompt_logprobs is not None or params.logprob_token_ids
            or request.lora_request is not None or request.resumable
            or request.use_structured_output):
        raise ValueError("Unsupported native decision request configuration")
    if any(t >= scheduler.vllm_config.model_config.get_vocab_size()
           for t in candidates(params)):
        raise ValueError("Native decision candidate outside model vocabulary")


def choose_step(scheduler):
    """Alternate score/generation steps when both classes have pending work.

Final score prefills already in flight are excluded: they need no further
forward. Blocked waiting requests retain the scheduler's promotion logic.
"""
    kinds = set()
    for request in (*scheduler.running, *scheduler.waiting, *scheduler.skipped_waiting):
        score = is_score(request)
        if (score and request in scheduler.running
                and request.num_computed_tokens >= request.num_prompt_tokens):
            continue
        kinds.add(score)
    if kinds == {True, False}:
        chosen = not getattr(scheduler, "_forjev_previous_step", False)
    else:
        chosen = kinds == {True}
    scheduler._forjev_previous_step = chosen
    return chosen


def remember_request(runner, req_id, params):
    requests = getattr(runner, "_forjev_requests", None)
    if requests is None:
        requests = runner._forjev_requests = {}
    ids = candidates(params)
    if ids is not None:
        requests[req_id] = list(ids)


def forget_request(runner, req_id):
    getattr(runner, "_forjev_requests", {}).pop(req_id, None)


def score_batch(runner, batch, hidden_states, finished_req_ids,
                ec_connector_output, boundary_logits_only):
    """Return None for generation; score homogeneous prefills before sampling."""
    ids_by_request = getattr(runner, "_forjev_requests", {})
    flags = [r in ids_by_request for r in batch.req_ids]
    if not any(flags):
        return None
    if not all(flags) or batch.num_draft_tokens:
        raise RuntimeError("Native decision batch isolation was violated")
    import torch
    from vllm.v1.outputs import ModelRunnerOutput

    # The checked V2 runner normally uses the same hidden row indices in sample().
    # No sampler, penalty, vocabulary filter or temperature has touched them.
    complete_rows = [i for i in range(batch.num_reqs)
                     if (boundary_logits_only or
                         int(batch.num_computed_prefill_tokens_np[i])
                         + int(batch.num_scheduled_tokens[i]) >= int(batch.prefill_len_np[i]))]
    rows = [None] * batch.num_reqs
    logits = None
    if complete_rows:
        complete_indices = torch.tensor(complete_rows, device=hidden_states.device,
                                        dtype=torch.long)
        indices = batch.logits_indices.index_select(0, complete_indices)
        logits = runner.model.compute_logits(hidden_states[indices])
        if logits is None or logits.ndim != 2 or logits.shape[0] != len(complete_rows):
            raise RuntimeError("Native decision logits are not aligned with request rows")
    for row_index, i in enumerate(complete_rows):
        req_id = batch.req_ids[i]
        columns = torch.tensor(ids_by_request[req_id], device=logits.device,
                               dtype=torch.long)
        # Copy selected columns only. A full-vocabulary logsumexp is unnecessary
        # for the conditional distribution over actions and is deliberately omitted.
        row = logits[row_index].index_select(0, columns).float().cpu()
        # Numerical failures travel to the request-local API validator; raising
        # here would turn a bad readout into an engine-wide worker failure.
        rows[i] = row

    if not boundary_logits_only:
        runner.postprocess_num_computed_tokens(batch)
        # Zero sampled outputs retains Mamba's neutral acceptance count (one).
        # Align mode needs the advanced GPU counts for recurrent-cache postprocess.
        runner.model_state.postprocess_state(
            batch.idx_mapping, 0, runner.req_states.num_computed_tokens.gpu)
    if runner.num_speculative_steps > 0:
        # Clear the transport's previous draft result; no drafter is invoked.
        runner.draft_tokens_handler.set_draft_tokens(
            batch, runner.req_states.draft_tokens[batch.idx_mapping, :0])
    output = ModelRunnerOutput(
        req_ids=list(batch.req_ids),
        req_id_to_index={r: i for i, r in enumerate(batch.req_ids)},
        sampled_token_ids=[[] for _ in batch.req_ids],
        pooler_output=rows,
        kv_connector_output=runner.kv_connector.post_forward(finished_req_ids),
        ec_connector_output=ec_connector_output,
    )
    return output


def numeric_output(state, pooling_output, finished, finish_reason, stop_reason):
    if not getattr(state, "_forjev_score", False):
        return None
    if not finished:
        raise RuntimeError("Native decision numeric output must finish the request")
    result = state._new_request_output(
        state.external_req_id,
        [state._new_completion_output([], finish_reason, stop_reason)], True)
    result.forjev_scores = pooling_output.tolist()
    return result


async def _final_output(generator, disconnected):
    async def collect():
        result = None
        async for output in generator:
            if result is not None:
                raise HTTPException(502, "Multiple native decision outputs")
            result = output
        return result

    async def watch_disconnect():
        while not await disconnected():
            await asyncio.sleep(0.1)

    reader = asyncio.create_task(collect())
    watcher = asyncio.create_task(watch_disconnect())
    try:
        done, _ = await asyncio.wait((reader, watcher), return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            await watcher  # propagate errors from the disconnect check
            raise asyncio.CancelledError()
        return await reader
    finally:
        reader.cancel()
        watcher.cancel()
        await asyncio.gather(reader, watcher, return_exceptions=True)


async def prefill_scores(chat, prompt, token_ids, *, request_id, disconnected,
                         skip_reading_prefix_cache=False):
    """Consume the patched engine's numeric output, requiring zero output tokens."""
    engine = chat.engine_client
    try:
        check_config(engine.vllm_config)
    except ValueError as exc:
        raise HTTPException(501, str(exc)) from exc
    from vllm.sampling_params import SamplingParams, RequestOutputKind
    params = SamplingParams(max_tokens=1, n=1, temperature=0,
                            detokenize=False, ignore_eos=True,
                            output_kind=RequestOutputKind.FINAL_ONLY,
                            skip_reading_prefix_cache=skip_reading_prefix_cache,
                            extra_args={MARKER: list(token_ids)})
    generator = engine.generate(prompt, params, request_id)
    completed = False
    try:
        result = await _final_output(generator, disconnected)
        completed = True
    finally:
        if not completed:
            await engine.abort(request_id)
        await generator.aclose()
    if (result is None or not result.finished or len(result.outputs) != 1
            or result.outputs[0].token_ids or result.outputs[0].finish_reason == "error"
            or not hasattr(result, "forjev_scores")):
        raise HTTPException(502, "Native decision output is missing or contains generated tokens")
    try:
        return score_response(token_ids, result.forjev_scores,
                              len(result.prompt_token_ids),
                              score_type="raw_logits", execution="prefill_logits")
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(502, "Invalid native candidate logits") from exc
