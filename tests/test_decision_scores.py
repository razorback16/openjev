import math

import pytest

from openjev.decision_scores import normalize, parse_response, score_response


def test_logits_and_full_vocabulary_logprobs_have_same_conditional_distribution():
    logits = [-10000.0, -9998.0, -9999.0]
    # Vocabulary also includes two tokens outside the candidate set.
    all_logits = logits + [-9997.0, -9996.0]
    peak = max(all_logits)
    log_z = peak + math.log(sum(math.exp(x - peak) for x in all_logits))
    raw = score_response([10, 11, 12], logits, 42, score_type="raw_logits",
                         execution="prefill_logits", log_partition=log_z)
    legacy = score_response([10, 11, 12], [x - log_z for x in logits], 42,
                            score_type="raw_logprobs", execution="engine_logprobs")
    assert raw["probabilities"] == pytest.approx(legacy["probabilities"])
    assert raw["candidate_mass"] == pytest.approx(legacy["candidate_mass"])
    assert 0 < raw["candidate_mass"] < 0.2
    assert raw["generated_tokens"] == 0
    assert legacy["generated_tokens"] == 1


def test_reordered_ids_are_mapped_to_request_order():
    data = score_response([20, 10], [1.0, 4.0], 17,
                          score_type="raw_logits", execution="prefill_logits")
    probabilities, tokens = parse_response(data, [10, 20], require_prefill=True)
    assert probabilities == pytest.approx(normalize([4.0, 1.0]))
    assert tokens == 17


@pytest.mark.parametrize("field,value", [
    ("schema", "other"), ("token_ids", [10, 10]), ("token_ids", [10]),
    ("token_ids", [10, 30]), ("token_ids", [True, 20]),
    ("scores", [float("nan"), 1.0]), ("scores", [float("inf"), 1.0]),
    ("scores", [0.0]), ("scores", [True, 1.0]),
    ("execution", "unknown"), ("generated_tokens", 1),
    ("score_type", "processed_logits"), ("usage", {"prompt_tokens": -1}),
])
def test_malformed_response_fails_closed(field, value):
    data = score_response([10, 20], [1.0, 4.0], 17,
                          score_type="raw_logits", execution="prefill_logits")
    data[field] = value
    with pytest.raises((ValueError, TypeError)):
        parse_response(data, [10, 20])


def test_prefill_mode_never_silently_downgrades_to_internal_generation():
    data = score_response([10, 20], [-1.0, -2.0], 17,
                          score_type="raw_logprobs", execution="engine_logprobs")
    with pytest.raises(ValueError, match="prefill-only"):
        parse_response(data, [10, 20], require_prefill=True)


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")]])
def test_invalid_scores_are_not_replaced_with_probabilities(values):
    with pytest.raises(ValueError):
        normalize(values)


def test_optional_vocabulary_mass_needs_full_vocab_normalization():
    data = score_response([10, 20], [100.0, 101.0], 17,
                          score_type="raw_logits", execution="prefill_logits")
    assert data["candidate_mass"] is None
    assert sum(data["probabilities"]) == pytest.approx(1)
    with pytest.raises(ValueError, match="mass"):
        score_response([10, 20], [0.0, 0.0], 17,
                       score_type="raw_logprobs", execution="engine_logprobs")
