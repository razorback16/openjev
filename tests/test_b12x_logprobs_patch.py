"""Run the actual patched B12X rejection methods with CPU tensor inputs.

Verification/GPU kernels are substituted; request/chunk selection and candidate
readout calls are the real supplied source, not a reimplementation of the patch.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from openjev import b12x_logprobs_patch as patch
from openjev import b12x_patch as native


NAME = "v1/worker/gpu/spec_decode/rejection_sampler.py"
SOURCE = Path(__file__).with_name("fixtures") / "b12x_rejection_sampler.py"


def methods(torch):
    source = patch.transform(NAME, SOURCE.read_text())
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
               and n.name == "RejectionSampler")
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
            and n.name in {"_get_logprobs_tensors", "_verify_in_chunks", "__call__"}]
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0),
                           ast.ClassDef(name="Extracted", bases=[], keywords=[], body=body,
                                        decorator_list=[])], type_ignores=[])
    calls = []
    class Flatten:
        def __getitem__(self, grid):
            def run(out, sampled, stride, counts, cu, **kwargs):
                for i in range(len(counts)):
                    start = int(cu[i]); n = int(counts[i])
                    out[start:start+n] = sampled[i, :n]
            return run
    def topk(logits, n, sampled, cu, **kwargs):
        mapping = kwargs["expanded_idx_mapping"]
        state = kwargs["logprob_token_ids_state"]
        width = max(n, kwargs["max_per_req_token_ids"])
        ids = torch.zeros((len(logits), 1+width), dtype=torch.long)
        ids[:, 0] = sampled
        for i in range(len(logits)):
            requested = state.ids.get(int(mapping[i]), [])
            columns = requested or torch.topk(logits[i], n).indices.tolist()
            ids[i, 1:1+len(columns)] = torch.tensor(columns)
        scores = torch.log_softmax(logits, -1).gather(-1, ids)
        result = NS(ids=ids, scores=scores, cu=cu)
        calls.append((mapping.tolist(), cu, result))
        return result
    class Tensors:
        @staticmethod
        def cat(rows, cu_num_generated_tokens=None):
            return NS(ids=torch.cat([r.ids for r in rows]),
                      scores=torch.cat([r.scores for r in rows]),
                      cu=cu_num_generated_tokens)
    globals = dict(torch=torch, np=np, NO_LOGPROBS=-1,
                   _flatten_sampled_kernel=Flatten(), compute_topk_scores=topk,
                   LogprobsTensors=Tensors, PROCESSED_LOGPROBS_MODES=set(),
                   get_max_chunk_logits=lambda _: 3,
                   get_num_nans=lambda _: None,
                   get_num_sampled_and_rejected=lambda counts, *args: (counts, torch.zeros_like(counts)),
                   SamplerOutput=NS)
    # The actual supplied chunk iterator is also used.
    iterator = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                    and n.name == "_iter_request_chunks")
    tree.body.insert(1, iterator)
    exec(compile(ast.fix_missing_locations(tree), str(SOURCE), "exec"), globals)
    obj = globals["Extracted"]()
    obj.enable_adaptive_verification = False
    state = NS(ids={3: [1, 4]}, max_num_token_ids=lambda _: 2)
    obj.sampler = NS(logprobs_mode="raw_logprobs", logprob_token_ids_state=state,
                     compute_nans=False, sampling_states=NS(max_num_logprobs=lambda _: 1),
                     req_states=NS(prefill_len=NS(gpu=torch.zeros(10))))
    def verify(logits, draft_logits, draft_sampled, pos, cu, idx, idx_np, expanded, local):
        counts = torch.tensor([int(cu[i+1]-cu[i]) for i in range(len(idx))])
        sampled = torch.zeros((len(idx), 3), dtype=torch.long)
        for i, n in enumerate(counts):
            sampled[i, :n] = torch.argmax(logits[int(cu[i]):int(cu[i+1])], -1)
        return logits, sampled, counts
    obj._verify = verify
    return obj, calls


@pytest.mark.parametrize("limit", [3, 99])
def test_mixed_planner_and_score_candidates_survive_request_chunks(limit):
    torch = pytest.importorskip("torch")
    obj, calls = methods(torch)
    logits = torch.tensor([[9., -5., 8., 7., -6.]] * 4)
    batch = NS(num_reqs=2, cu_num_logits_np=np.array([0, 3, 4]),
               cu_num_logits=torch.tensor([0, 3, 4]), idx_mapping=torch.tensor([9, 3]),
               idx_mapping_np=np.array([9, 3]), expanded_idx_mapping=torch.tensor([9, 9, 9, 3]),
               expanded_local_pos=torch.tensor([0, 1, 2, 0]))
    sampled, counts, result = obj._verify_in_chunks(
        logits, batch, None, torch.zeros(4, dtype=torch.long), torch.zeros(4), limit, 1, 2)
    assert sampled.tolist() == [[0, 0, 0], [0, 0, 0]]
    assert counts.tolist() == [3, 1]  # verification/acceptance is unchanged
    assert result.cu == [0, 3, 4]
    assert result.ids[3].tolist() == [0, 1, 4]  # both candidates are outside top-k
    torch.testing.assert_close(result.scores[3, 1:], torch.log_softmax(logits[3], -1)[[1, 4]])
    assert result.ids[:3, 1].tolist() == [0, 0, 0]  # planner keeps ordinary top-k
    assert [i for mapping, _, _ in calls for i in mapping] == [9, 9, 9, 3]


def test_custom_only_logprobs_and_adaptive_boundaries():
    torch = pytest.importorskip("torch")
    obj, calls = methods(torch)
    obj.enable_adaptive_verification = True
    logits = torch.tensor([[5., -5., 0., 1., -6.]] * 3)
    cu = torch.tensor([0, 2, 3])
    out = obj._get_logprobs_tensors(torch.tensor([[0, 0], [0, 0]]),
        torch.tensor([2, 1]), logits, cu, np.array([0, 5, 6]), -1, 2,
        torch.tensor([9, 9, 3]))
    assert torch.equal(out.cu, cu) and out.cu is not cu
    assert out.ids[2].tolist() == [0, 1, 4]
    calls.clear()
    assert obj._get_logprobs_tensors(None, None, None, None, None, -1) is None
    assert calls == []  # requests without logprobs do no readout work


def test_entrypoint_discovers_candidates_instead_of_only_topk():
    torch = pytest.importorskip("torch")
    obj, calls = methods(torch)
    logits = torch.tensor([[9., -5., 8., 7., -6.]] * 4)
    obj.sampler.sampling_states.max_num_logprobs = lambda _: -1
    batch = NS(num_reqs=2, cu_num_logits_np=np.array([0, 3, 4]),
               cu_num_logits=torch.tensor([0, 3, 4]), idx_mapping=torch.tensor([9, 3]),
               idx_mapping_np=np.array([9, 3]), expanded_idx_mapping=torch.tensor([9, 9, 9, 3]),
               expanded_local_pos=torch.tensor([0, 1, 2, 0]),
               input_ids=torch.zeros(4, dtype=torch.long), logits_indices=torch.arange(4),
               positions=torch.arange(4), seq_lens=torch.tensor([20, 10]))
    out = obj(logits, batch)
    assert out.num_sampled.tolist() == [3, 1]
    assert out.logprobs_tensors.ids[3].tolist() == [0, 1, 4]


def test_checked_source_fixture_and_unknown_source_rejection():
    import hashlib, json
    manifest = json.loads(Path(patch.__file__).with_name("b12x_logprobs_manifest.json").read_text())
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == manifest["files"][NAME]["original"]
    changed = patch.transform(NAME, SOURCE.read_text())
    assert hashlib.sha256(changed.encode()).hexdigest() == manifest["files"][NAME]["patched"]
    with pytest.raises(ValueError, match="anchor"):
        patch.transform(NAME, "unrecognized source")


def test_existing_launcher_preflight_selects_bridge_image_profile(monkeypatch):
    monkeypatch.setenv("FORJEV_B12X_PROFILE", "bridge")
    monkeypatch.setattr(patch, "main", lambda: "bridge verifier")
    assert native.main() == "bridge verifier"


def test_bridge_installer_apply_verify_revert_without_native_scheduler(tmp_path):
    import shutil
    root = os.getenv("B12X_SOURCE_DIR")
    if not root:
        pytest.skip("B12X_SOURCE_DIR supplies original serving files")
    root = Path(root)
    for name in patch.RULES:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, dest)
    prepared = patch.prepare(tmp_path)
    assert len(prepared) == 3
    native.apply(prepared)
    assert patch.prepare(tmp_path) == []
    assert not (tmp_path / "v1/core/sched/scheduler.py").exists()
    native.apply(patch.prepare(tmp_path, revert=True), revert=True)
    assert len(patch.prepare(tmp_path)) == 3
    worker = tmp_path / "v1/worker/gpu/model_runner.py"
    worker.write_text(worker.read_text()+"# unknown edit\n")
    with pytest.raises(ValueError, match="fingerprint"):
        patch.prepare(tmp_path)
