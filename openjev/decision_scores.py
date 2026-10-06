"""Versioned numeric readout contract shared by ForJev and its vLLM route.

These scores are next-token probabilities, not calibrated success estimates.
No dependency on torch or vLLM in the HTTP adapter.
"""
import math


SCHEMA = "forjev.decision_scores.v1"
EXECUTIONS = {"engine_logprobs", "prefill_logits"}


def normalize(values):
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("candidate scores must be finite and nonempty")
    peak = max(values)
    weights = [math.exp(v - peak) for v in values]
    total = math.fsum(weights)
    return [w / total for w in weights]


def score_response(token_ids, values, prompt_tokens, *, score_type, execution,
                   log_partition=None):
    """Build a response from raw logits or raw vocabulary log probabilities.

For logits, an optional full-vocabulary logsumexp permits candidate_mass.
The conditional distribution only needs the selected columns.
"""
    if (not token_ids or len(token_ids) != len(values)
            or len(set(token_ids)) != len(token_ids)
            or any(type(t) is not int or t < 0 for t in token_ids)):
        raise ValueError("invalid candidate token IDs")
    if type(prompt_tokens) is not int or prompt_tokens < 0:
        raise ValueError("invalid prompt token count")
    if score_type not in {"raw_logits", "raw_logprobs"} or execution not in EXECUTIONS:
        raise ValueError("unsupported score semantics")
    values = [float(v) for v in values]
    probabilities = normalize(values)
    logprobs = values if score_type == "raw_logprobs" else None
    if log_partition is not None:
        if score_type != "raw_logits" or not math.isfinite(log_partition):
            raise ValueError("invalid vocabulary log partition")
        logprobs = [v - log_partition for v in values]
    mass = None
    if logprobs is not None:
        if any(v > 1e-5 for v in logprobs):
            raise ValueError("invalid vocabulary log probabilities")
        mass = math.fsum(math.exp(v) for v in logprobs)
        if mass > 1 + 1e-5:
            raise ValueError("candidate probability mass exceeds one")
        mass = min(mass, 1.0)
    return {
        "schema": SCHEMA,
        "score_type": score_type,
        "execution": execution,
        "token_ids": list(token_ids),
        "scores": values,
        "probabilities": probabilities,
        "candidate_mass": mass,
        # engine_logprobs still runs a one-token internal generation request.
        "generated_tokens": 0 if execution == "prefill_logits" else 1,
        "usage": {"prompt_tokens": prompt_tokens},
    }


def parse_response(data, requested_ids, *, require_prefill=False):
    """Reject incomplete, duplicate or ambiguous readouts; never fill gaps."""
    if data["schema"] != SCHEMA:
        raise ValueError("unsupported scoring schema")
    ids = data["token_ids"]
    if (len(ids) != len(requested_ids) or len(set(ids)) != len(ids)
            or any(type(t) is not int or t < 0 for t in ids)
            or set(ids) != set(requested_ids)):
        raise ValueError("incomplete candidate token IDs")
    execution = data["execution"]
    if execution not in EXECUTIONS:
        raise ValueError("unsupported execution")
    generated = data["generated_tokens"]
    if type(generated) is not int or generated != (0 if execution == "prefill_logits" else 1):
        raise ValueError("invalid execution metadata")
    if require_prefill and execution != "prefill_logits":
        raise ValueError("prefill-only scoring is not installed upstream")
    if data["score_type"] not in {"raw_logits", "raw_logprobs"}:
        raise ValueError("only unprocessed model scores are supported")
    scores = data["scores"]
    if len(scores) != len(ids) or any(type(v) not in (int, float) for v in scores):
        raise ValueError("invalid candidate scores")
    mapped = dict(zip(ids, scores))
    probabilities = normalize([mapped[t] for t in requested_ids])
    tokens = data["usage"]["prompt_tokens"]
    if type(tokens) is not int or tokens < 0:
        raise ValueError("invalid prompt token count")
    return probabilities, tokens
