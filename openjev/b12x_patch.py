"""Apply/revert a source-fingerprinted B12X integration. Defaults to dry run.

Run inside the vLLM image with the installed openjev package. All input files
are checked and compiled before any file is written. Backups are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


RULES = {
    "v1/core/sched/scheduler.py": [
        ("        remaining = request.num_tokens - num_computed_tokens - num_new_tokens\n",
         "        from openjev.b12x_runtime import is_score\n        if is_score(request):\n            return max(num_new_tokens, 0)\n        remaining = request.num_tokens - num_computed_tokens - num_new_tokens\n"),
        ("        self.current_step += 1\n",
         "        self.current_step += 1\n        from openjev.b12x_runtime import choose_step, is_score\n        forjev_step = choose_step(self)\n"),
        ("        if spec is not None:\n            if spec.use_dflash():",
         "        if spec is not None and not forjev_step:\n            if spec.use_dflash():"),
        ("if spec is not None and separate_draft_input_tokens == 0",
         "if spec is not None and not forjev_step and separate_draft_input_tokens == 0"),
        ("if self.max_parallel_prefills > 1 and has_possible_prefill\n",
         "if self.max_parallel_prefills > 1 and has_possible_prefill and not forjev_step\n"),
        ("        if self.compute_share_controller is not None:\n            prior_contention",
         "        if self.compute_share_controller is not None and not forjev_step:\n            prior_contention"),
        ("        defer_prefills = legacy_defer_prefills or adaptive_defer_prefills\n",
         "        defer_prefills = (legacy_defer_prefills or adaptive_defer_prefills) and not forjev_step\n"),
        ("                request = self.running[req_index]\n",
         "                request = self.running[req_index]\n                if is_score(request) != forjev_step:\n                    req_index += 1\n                    continue\n"),
        ("        reserve_boundary_logits_step = self._has_waiting_boundary_logits()\n",
         "        reserve_boundary_logits_step = self._has_waiting_boundary_logits()\n        if reserve_boundary_logits_step:\n            reserve_boundary_logits_step = (is_score(self._select_waiting_queue_for_scheduling().peek_request()) == forjev_step)\n"),
        ("                request_id = request.request_id\n\n                # try to promote",
         "                request_id = request.request_id\n                if is_score(request) != forjev_step:\n                    request_queue.remove_request(request)\n                    if prefill_interleave_step is not None:\n                        prefill_interleave_step.mark_unavailable(request_id)\n                    step_skipped_waiting.prepend_request(request)\n                    continue\n\n                # try to promote"),
        ("                            num_lookahead_tokens=self.num_lookahead_tokens,\n",
         "                            num_lookahead_tokens=0 if forjev_step else self.num_lookahead_tokens,\n"),
        ("                    0 if limit_lookahead_tokens else self.num_lookahead_tokens\n",
         "                    0 if (limit_lookahead_tokens or forjev_step) else self.num_lookahead_tokens\n"),
        ("        scheduled_encoder_input_stats = None\n",
         "        if forjev_step:\n            num_spec_tokens_to_schedule = 0\n\n        scheduled_encoder_input_stats = None\n"),
        ("        existing = self.requests.get(request.request_id)\n",
         "        from openjev.b12x_runtime import validate_request\n        validate_request(self, request)\n        existing = self.requests.get(request.request_id)\n"),
        ("            # Drop-mode stale output (same-step resume) is discarded entirely.\n",
         "            from openjev.b12x_runtime import is_score\n            if output_is_stale and is_score(request):\n                continue\n\n            # Drop-mode stale output (same-step resume) is discarded entirely.\n"),
        ("            elif request.pooling_params and pooler_output is not None:\n",
         "            elif (request.pooling_params or is_score(request)) and pooler_output is not None:\n"),
        ("        if self.acceptance_length_controller is not None and batch_size > 0:\n            update =",
         "        if (self.acceptance_length_controller is not None and batch_size > 0\n                and scheduler_output.num_spec_tokens_to_schedule != 0):\n            update ="),
    ],
    "v1/core/sched/async_scheduler.py": [
        ("            # Add placeholders for the new draft/spec tokens.\n",
         "            from openjev.b12x_runtime import is_score\n            if is_score(request):\n                # Reserve one lifecycle placeholder, but never speculative IDs.\n                # This fences the final prefill until its numeric output arrives.\n                request.spec_token_ids = []\n                continue\n            # Add placeholders for the new draft/spec tokens.\n"),
    ],
    "v1/worker/gpu/model_runner.py": [
        ("        # Call model_state.remove_request *before* req_states.remove_request\n",
         "        from openjev.b12x_runtime import forget_request\n        forget_request(self, req_id)\n        # Call model_state.remove_request *before* req_states.remove_request\n"),
        ("            sampling_params = new_req_data.sampling_params\n",
         "            sampling_params = new_req_data.sampling_params\n            from openjev.b12x_runtime import remember_request\n            remember_request(self, req_id, sampling_params)\n"),
        ("                    checkpoint = new_req.boundary_checkpoint\n",
         "                    from openjev.b12x_runtime import candidates\n                    if candidates(new_req.sampling_params) is not None:\n                        continue\n                    checkpoint = new_req.boundary_checkpoint\n"),
        ("        sampler_output, num_sampled, num_rejected = self.sample(\n",
         "        from openjev.b12x_runtime import score_batch\n        forjev_output = score_batch(self, input_batch, hidden_states, finished_req_ids,\n                                    ec_connector_output, boundary_logits_only)\n        if forjev_output is not None:\n            return forjev_output\n\n        sampler_output, num_sampled, num_rejected = self.sample(\n"),
    ],
    "v1/engine/output_processor.py": [
        ("        return cls(\n            request_id=request.request_id,\n",
         "        state = cls(\n            request_id=request.request_id,\n"),
        ("            stream_input=request.resumable,\n        )\n",
         "            stream_input=request.resumable,\n        )\n        from openjev.b12x_runtime import candidates\n        state._forjev_score = candidates(request.sampling_params) is not None\n        return state\n"),
        ("        if pooling_output is not None:\n            return self._new_request_output(\n",
         "        if pooling_output is not None:\n            from openjev.b12x_runtime import numeric_output\n            numeric = numeric_output(self, pooling_output, finished, finish_reason, stop_reason)\n            if numeric is not None:\n                return numeric\n            return self._new_request_output(\n"),
    ],
    "entrypoints/launchers/app.py": [
        ("    init_exception_handler(app)\n",
         "    import os\n    if os.environ.get('FORJEV_NATIVE_SCORES', '1') != '0' and 'generate' in supported_tasks:\n        from openjev.vllm_scores import install_routes\n        from openjev.b12x_runtime import prefill_scores\n        install_routes(app, prefill_score=prefill_scores)\n\n    init_exception_handler(app)\n"),
    ],
}


def transform(path, source):
    for old, new in RULES[path]:
        if source.count(old) != 1:
            raise ValueError(f"{path}: expected one patch anchor, found {source.count(old)}")
        source = source.replace(old, new, 1)
    compile(source, path, "exec")
    return source


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, data):
    mode = path.stat().st_mode if path.exists() else 0o644
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temp_path = Path(tmp.name)
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
    try:
        os.chmod(temp_path, mode)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def prepare(root, *, revert=False, manifest_name="b12x_manifest.json",
            rules=RULES, transform_source=transform,
            backup_suffix=".forjev-original"):
    manifest = json.loads(Path(__file__).with_name(manifest_name).read_text())
    prepared = []
    for name in rules:
        path = root / name
        data = path.read_bytes()
        backup = path.with_name(path.name + backup_suffix)
        expected = manifest["files"][name]
        digest = sha(data)
        if digest not in (expected["original"], expected["patched"]):
            raise ValueError(f"{name}: source fingerprint mismatch; refusing to patch")
        if revert:
            if digest == expected["original"]:
                continue
            original = backup.read_bytes()
            if sha(original) != expected["original"]:
                raise ValueError(f"{name}: backup fingerprint mismatch")
            prepared.append((path, data, original, backup))
        elif digest == expected["original"]:
            if backup.exists() and sha(backup.read_bytes()) != digest:
                raise ValueError(f"{name}: existing backup does not match source")
            patched = transform_source(name, data.decode()).encode()
            if sha(patched) != expected["patched"]:
                raise ValueError(f"{name}: patch fingerprint mismatch")
            prepared.append((path, data, patched, backup))
    return prepared


def apply(prepared, *, revert=False):
    written = []
    try:
        for path, before, after, backup in prepared:
            if path.read_bytes() != before:
                raise ValueError(f"{path.name}: source changed after verification")
            if not revert and not backup.exists():
                atomic_write(backup, before)
            atomic_write(path, after)
            written.append((path, before))
    except BaseException:
        for path, before in reversed(written):
            atomic_write(path, before)
        raise


def main():
    # The bridge image uses the same launcher preflight command, while retaining
    # a separate manifest/installer and no native scheduler hooks.
    if os.environ.get("FORJEV_B12X_PROFILE") == "bridge":
        from .b12x_logprobs_patch import main as bridge_main
        return bridge_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Installed vllm package directory")
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument("--apply", action="store_true")
    flags.add_argument("--revert", action="store_true")
    args = parser.parse_args()
    root = args.root
    if root is None:
        import vllm
        root = Path(vllm.__file__).parent
    prepared = prepare(root, revert=args.revert)
    if args.apply or args.revert:
        apply(prepared, revert=args.revert)
    action = "reverted" if args.revert else "patched" if args.apply else "verified (dry run)"
    print(f"B12X ForJev: {len(prepared)} files {action}; restart required after writes")


if __name__ == "__main__":
    main()
