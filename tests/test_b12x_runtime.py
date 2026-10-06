import asyncio
import ast
import copy
import os
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from openjev import b12x_runtime as runtime
from openjev import b12x_patch as patch


def config(**changes):
    return NS(use_v2_model_runner=True, parallel_config=NS(),
              model_config=NS(get_vocab_size=lambda: 100), **changes)


def params(ids=None):
    return NS(extra_args={} if ids is None else {runtime.MARKER: ids},
              max_tokens=1, n=1, logprobs=None, prompt_logprobs=None,
              logprob_token_ids=None)


def request(ids=None, computed=0):
    return NS(sampling_params=params(ids), num_computed_tokens=computed,
              num_prompt_tokens=5, lora_request=None, resumable=False,
              use_structured_output=False)


@pytest.mark.parametrize("ids", [[1, 1], [True, 2], [1], [1, -2], "ab", [1] * 256])
def test_invalid_native_marker(ids):
    with pytest.raises(ValueError):
        runtime.candidates(params(ids))


def test_batch_selection_fairness_and_final_prefill_fence():
    score, planner = request([10, 20]), request()
    sched = NS(running=[score, planner], waiting=[], skipped_waiting=[])
    assert [runtime.choose_step(sched) for _ in range(6)] == [True, False] * 3
    score.num_computed_tokens = 5
    assert runtime.choose_step(sched) is False
    sched.running = [score]
    assert runtime.choose_step(sched) is False
    sched.waiting = [request([10, 20])]
    assert runtime.choose_step(sched) is True


@pytest.mark.parametrize("change", ["v1", "tp", "kv", "encoder", "pooling", "diffusion", "encoder_only"])
def test_unsupported_runtime_is_rejected(change):
    cfg = config()
    if change == "v1":
        cfg.use_v2_model_runner = False
    elif change == "tp":
        cfg.parallel_config.tensor_parallel_size = 2
    elif change == "kv":
        cfg.kv_transfer_config = object()
    elif change == "encoder":
        cfg.model_config.is_encoder_decoder = True
    elif change == "diffusion":
        cfg.model_config.is_diffusion = True
    elif change == "encoder_only":
        cfg.is_mm_encoder_only = True
    else:
        cfg.model_config.runner_type = "pooling"
    with pytest.raises(ValueError):
        runtime.check_config(cfg)


def test_admission_rejects_unhandled_request_modes():
    sched = NS(vllm_config=config())
    runtime.validate_request(sched, request([10, 20]))
    for name in ("resumable", "use_structured_output", "lora_request"):
        req = request([10, 20])
        setattr(req, name, True)
        with pytest.raises(ValueError):
            runtime.validate_request(sched, req)
    with pytest.raises(ValueError):
        runtime.validate_request(sched, request([10, 100]))


@pytest.fixture
def torch_outputs(monkeypatch):
    torch = pytest.importorskip("torch")
    outputs = ModuleType("vllm.v1.outputs")
    outputs.ModelRunnerOutput = NS
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.outputs", outputs)
    return torch


def test_native_chunk_advances_counts_before_recurrent_cache_postprocess(monkeypatch):
    # No lm_head work is needed for an incomplete chunk; test the lifecycle
    # without CUDA/torch and require the count buffer needed by Mamba align.
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    outputs = ModuleType("vllm.v1.outputs")
    outputs.ModelRunnerOutput = NS
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.outputs", outputs)
    counts = [0]
    checkpoints = []
    def advance(batch):
        counts[0] += batch.num_scheduled_tokens[0]
    def postprocess(idx, sampled, computed):
        assert computed is counts and sampled == 0
        checkpoints.append(computed[0])
    runner = NS(_forjev_requests={"q": [1, 2]},
                req_states=NS(num_computed_tokens=NS(gpu=counts)),
                postprocess_num_computed_tokens=advance,
                model_state=NS(postprocess_state=postprocess),
                num_speculative_steps=0, kv_connector=NS(post_forward=lambda _: None))
    batch = NS(req_ids=["q"], num_reqs=1, num_draft_tokens=0,
               num_computed_prefill_tokens_np=[0], num_scheduled_tokens=[2],
               prefill_len_np=[6], idx_mapping=[0])
    output = runtime.score_batch(runner, batch, None, set(), None, False)
    assert output.pooler_output == [None] and output.sampled_token_ids == [[]]
    assert checkpoints == [2]


def runner_and_batch(torch, *, spec=True):
    events = []
    # The target's actual lm_head matrix is used by compute_logits. Each hidden
    # row has a distinct signature to detect column/row and compaction mistakes.
    head = torch.arange(400, dtype=torch.float32).reshape(4, 100) / 100
    runner = NS(
        model=NS(compute_logits=lambda hidden: hidden @ head),
        postprocess_num_computed_tokens=lambda b: events.append("computed"),
        model_state=NS(postprocess_state=lambda idx, count, computed: events.append(("state", count))),
        kv_connector=NS(post_forward=lambda ids: ("kv", ids)),
        num_speculative_steps=3 if spec else 0,
        draft_tokens_handler=NS(set_draft_tokens=lambda b, d: events.append(("draft", d.shape[1]))),
        req_states=NS(draft_tokens=torch.ones((9, 3), dtype=torch.long),
                      num_computed_tokens=NS(gpu=torch.zeros(9, dtype=torch.int32))),
    )
    runtime.remember_request(runner, "second", params([20, 10]))
    runtime.remember_request(runner, "first", params([10, 30]))
    batch = NS(req_ids=["second", "first"], num_reqs=2, num_draft_tokens=0,
               logits_indices=torch.tensor([3, 1]), idx_mapping=torch.tensor([7, 2]),
               num_computed_prefill_tokens_np=np.array([3, 0]),
               num_scheduled_tokens=np.array([2, 2]), prefill_len_np=np.array([5, 4]))
    hidden = torch.eye(4)
    return runner, batch, hidden, events, head


def test_native_projection_chunking_request_order_and_no_tokens(torch_outputs):
    torch = torch_outputs
    runner, batch, hidden, events, head = runner_and_batch(torch)
    output = runtime.score_batch(runner, batch, hidden, {"prior"}, "ec", False)
    assert output.req_ids == ["second", "first"]
    assert output.req_id_to_index == {"second": 0, "first": 1}
    torch.testing.assert_close(output.pooler_output[0], head[3, [20, 10]])
    assert output.pooler_output[1] is None  # incomplete chunk emits no result
    assert output.sampled_token_ids == [[], []]
    assert output.kv_connector_output == ("kv", {"prior"})
    assert output.ec_connector_output == "ec"
    assert events == ["computed", ("state", 0), ("draft", 0)]
    # Different token labels/request mapping remain intact across the next chunk.
    batch.num_computed_prefill_tokens_np[1] = 2
    output = runtime.score_batch(runner, batch, hidden, set(), None, False)
    torch.testing.assert_close(output.pooler_output[1], head[1, [10, 30]])


def test_saved_boundary_logits_skip_state_advancement(torch_outputs):
    runner, batch, hidden, events, head = runner_and_batch(torch_outputs, spec=False)
    output = runtime.score_batch(runner, batch, hidden, set(), None, True)
    assert all(row is not None for row in output.pooler_output)
    assert events == []


def test_mixed_batch_fails_before_lm_head_and_sampling(torch_outputs):
    runner, batch, hidden, events, _ = runner_and_batch(torch_outputs)
    runtime.forget_request(runner, "first")
    with pytest.raises(RuntimeError, match="isolation"):
        runtime.score_batch(runner, batch, hidden, set(), None, False)
    runtime.forget_request(runner, "second")
    assert runtime.score_batch(runner, batch, hidden, set(), None, False) is None
    assert events == []


def test_nonfinite_logits_reach_request_validator_without_crashing_worker(torch_outputs):
    runner, batch, hidden, _, _ = runner_and_batch(torch_outputs)
    hidden[3, 0] = float("nan")
    output = runtime.score_batch(runner, batch, hidden, set(), None, False)
    assert not torch_outputs.isfinite(output.pooler_output[0]).all()


def test_incomplete_prefill_does_not_project_lm_head(torch_outputs):
    runner, batch, hidden, events, _ = runner_and_batch(torch_outputs)
    batch.num_computed_prefill_tokens_np[:] = 0
    runner.model.compute_logits = lambda h: pytest.fail("no final row to project")
    output = runtime.score_batch(runner, batch, hidden, set(), None, False)
    assert output.pooler_output == [None, None]
    assert output.sampled_token_ids == [[], []]


@pytest.fixture
def source_root():
    directory = os.environ.get("B12X_SOURCE_DIR")
    if not directory:
        pytest.skip("Set B12X_SOURCE_DIR to run tests against the uploaded runtime")
    return Path(directory)


def extract_method(source, class_name, method_name, globals):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                               and n.name == method_name))
    method.decorator_list = []
    # Compile the REAL supplied method, with deferred annotations and stubs only
    # for the surrounding runtime. This tests insertion and engine conventions.
    body = method
    if "__class__" in globals:
        globals["_base"] = globals["__class__"].__bases__[0]
        body = ast.ClassDef(name="_Extracted", bases=[ast.Name(id="_base", ctx=ast.Load())],
                            keywords=[], body=[method], decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), body], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "actual-b12x-method", "exec"), globals)
    return getattr(globals["_Extracted"], method_name) if "__class__" in globals else globals[method_name]


def test_real_async_scheduler_fences_score_without_draft_ids(source_root):
    name = "v1/core/sched/async_scheduler.py"
    source = patch.transform(name, (source_root / name).read_text())
    class Scheduler:
        def _update_after_schedule(self, out):
            pass
    class AsyncScheduler(Scheduler):
        pass
    method = extract_method(source, "AsyncScheduler", "_update_after_schedule",
                            {"__class__": AsyncScheduler})
    req = request([10, 20])
    req.is_prefill_chunk = False
    req.num_output_placeholders = 0
    req.spec_token_ids = []
    sched = method.__globals__["_Extracted"]()
    sched.num_spec_tokens = 3
    sched.num_sampled_tokens_per_step = 1
    sched.requests = {"score": req}
    sched.use_v2_model_runner = True
    out = NS(scheduled_spec_decode_tokens={}, num_scheduled_tokens={"score": 5},
             pending_structured_output_tokens=False,
             resolve_num_spec_tokens_to_schedule=lambda _: 0)
    method(sched, out)
    assert req.num_output_placeholders == 1
    assert req.spec_token_ids == []


def test_real_output_processor_returns_numeric_generation_output(source_root, torch_outputs):
    name = "v1/engine/output_processor.py"
    source = patch.transform(name, (source_root / name).read_text())
    method = extract_method(source, "RequestState", "make_request_output", {
        "RequestOutputKind": NS(FINAL_ONLY="final", DELTA="delta")})
    state = NS(output_kind="final", stream_interval=1, external_req_id="score",
               _forjev_score=True,
               _new_completion_output=lambda ids, finish, stop: NS(token_ids=ids),
               _new_request_output=lambda rid, outputs, finished: NS(
                   request_id=rid, outputs=outputs, finished=finished))
    result = method(state, [], torch_outputs.tensor([2.0, 4.0]), "stop", None)
    assert result.finished and result.outputs[0].token_ids == []
    assert result.forjev_scores == [2.0, 4.0]


def test_real_runner_sample_method_bypasses_sampler_and_drafter(source_root, torch_outputs):
    name = "v1/worker/gpu/model_runner.py"
    source = patch.transform(name, (source_root / name).read_text())
    method = extract_method(source, "GPUModelRunner", "sample_tokens", {
        "pcp": NS(maybe_restore_pcp_for_sampling=lambda manager, hidden, batch: (hidden, batch))})
    runner, batch, hidden, events, _ = runner_and_batch(torch_outputs)
    runner.execute_model_state = NS(
        input_batch=batch, attn_metadata=None, slot_mappings_by_layer=None,
        hidden_states=hidden, aux_hidden_states=None, finished_req_ids=set(),
        ec_connector_output=None, routed_experts=None, num_spec_tokens_to_schedule=0,
        boundary_logits_only=False, boundary_aux_block_id=None)
    runner.is_last_pp_rank = True
    runner.pcp_manager = None
    runner.sample = lambda *args: pytest.fail("sampler must not run")
    runner.speculator = NS(propose=lambda *args: pytest.fail("MTP must not run"))
    result = method(runner, None)
    assert result.sampled_token_ids == [[], []]
    assert runner.execute_model_state is None
    assert events == ["computed", ("state", 0), ("draft", 0)]


def test_real_scheduler_does_not_reserve_mtp_prefill_tail(source_root):
    name = "v1/core/sched/scheduler.py"
    source = patch.transform(name, (source_root / name).read_text())
    method = extract_method(source, "Scheduler", "_reserve_prefill_lookahead", {})
    sched = NS(num_prefill_lookahead=3)
    score, planner = request([10, 20]), request()
    score.num_tokens = planner.num_tokens = 5
    # A tiny prefill budget must not leave a native score permanently waiting
    # for a drafter-tail reservation that it will never use.
    assert method(sched, score, 2, 1) == 1
    assert method(sched, planner, 2, 1) == 0


def test_source_patch_dry_run_apply_idempotence_and_rollback(source_root, tmp_path):
    for name in patch.RULES:
        dst = tmp_path / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / name, dst)
    prepared = patch.prepare(tmp_path)
    assert len(prepared) == 5
    patch.apply(prepared)
    assert patch.prepare(tmp_path) == []
    reverted = patch.prepare(tmp_path, revert=True)
    assert len(reverted) == 5
    patch.apply(reverted, revert=True)
    assert len(patch.prepare(tmp_path)) == 5
    path = tmp_path / next(iter(patch.RULES))
    path.write_text(path.read_text() + "\n# unexpected upstream update\n")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        patch.prepare(tmp_path)


def test_apply_failure_restores_files(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"original-a")
    b.write_bytes(b"original-b")
    original_write = patch.atomic_write
    def fail_on_second(path, data):
        if path == b and data == b"changed-b":
            raise OSError("simulated disk failure")
        original_write(path, data)
    monkeypatch.setattr(patch, "atomic_write", fail_on_second)
    with pytest.raises(OSError):
        patch.apply([(a, a.read_bytes(), b"changed-a", tmp_path / "a.backup"),
                     (b, b.read_bytes(), b"changed-b", tmp_path / "b.backup")])
    assert a.read_bytes() == b"original-a" and b.read_bytes() == b"original-b"


@pytest.mark.parametrize("defect", [None, "token", "missing", "incomplete", "nonfinite"])
def test_native_provider_contract_and_no_logprobs(monkeypatch, defect):
    module = ModuleType("vllm.sampling_params")
    module.SamplingParams = lambda **kw: NS(**kw)
    module.RequestOutputKind = NS(FINAL_ONLY="final")
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", module)
    class Engine:
        vllm_config = config()
        def generate(self, prompt, args, request_id):
            assert args.extra_args == {runtime.MARKER: [10, 20]}
            assert not hasattr(args, "logprobs")
            assert args.output_kind == "final" and args.detokenize is False
            async def outputs():
                result = NS(finished=defect != "incomplete", prompt_token_ids=[1, 2],
                            outputs=[NS(token_ids=[3] if defect == "token" else [],
                                        finish_reason="stop")])
                if defect != "missing":
                    result.forjev_scores = [2.0, float("nan") if defect == "nonfinite" else 4.0]
                yield result
            return outputs()
        async def abort(self, rid):
            pytest.fail("completed request")
    async def connected():
        return False
    async def run():
        return await runtime.prefill_scores(NS(engine_client=Engine()), {}, [10, 20],
                                           request_id="score", disconnected=connected)
    if defect:
        with pytest.raises(Exception) as error:
            asyncio.run(run())
        assert error.value.status_code == 502
    else:
        result = asyncio.run(run())
        assert result["generated_tokens"] == 0 and result["execution"] == "prefill_logits"
        assert result["probabilities"][1] > 0.88


def test_native_disconnect_aborts_before_first_output(monkeypatch):
    module = ModuleType("vllm.sampling_params")
    module.SamplingParams = lambda **kw: NS(**kw)
    module.RequestOutputKind = NS(FINAL_ONLY="final")
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", module)
    events = []
    class Engine:
        vllm_config = config()
        def generate(self, *args):
            async def results():
                try:
                    await asyncio.Event().wait()
                    yield None
                finally:
                    events.append("closed")
            return results()
        async def abort(self, rid):
            events.append(("abort", rid))
    async def disconnected():
        await asyncio.sleep(0)
        return True
    async def run():
        await runtime.prefill_scores(NS(engine_client=Engine()), {}, [10, 20],
                                     request_id="score", disconnected=disconnected)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(asyncio.wait_for(run(), timeout=1))
    assert events == ["closed", ("abort", "score")]
