"""Checked B12X bridge integration: preserve custom logprobs in MTP batches.

Does not change scheduler admission, batch composition, token verification or
output transport. Defaults to dry run; build from the original B12X image.
"""
import argparse
from pathlib import Path

from .b12x_patch import apply, prepare as prepare_sources


RULES = {
    "v1/worker/gpu/spec_decode/rejection_sampler.py": [
        ("        max_num_logprobs: int,\n    ) -> LogprobsTensors | None:\n"
         "        if max_num_logprobs == NO_LOGPROBS:\n            return None\n",
         "        max_num_logprobs: int,\n        max_per_req_token_ids: int = 0,\n"
         "        expanded_idx_mapping: torch.Tensor | None = None,\n"
         "    ) -> LogprobsTensors | None:\n"
         "        if max_num_logprobs == NO_LOGPROBS and max_per_req_token_ids == 0:\n"
         "            return None\n"
         "        if max_num_logprobs == NO_LOGPROBS:\n            max_num_logprobs = 0\n"),
        ("            cu_num_generated_tokens,\n            logits_mode=self.sampler.logprobs_mode\n",
         "            cu_num_generated_tokens,\n"
         "            logprob_token_ids_state=self.sampler.logprob_token_ids_state,\n"
         "            expanded_idx_mapping=expanded_idx_mapping,\n"
         "            max_per_req_token_ids=max_per_req_token_ids,\n"
         "            logits_mode=self.sampler.logprobs_mode\n"),
        ("        max_num_logprobs: int,\n    ) -> tuple[torch.Tensor, torch.Tensor, LogprobsTensors | None]:\n",
         "        max_num_logprobs: int,\n        max_per_req_token_ids: int = 0,\n"
         "    ) -> tuple[torch.Tensor, torch.Tensor, LogprobsTensors | None]:\n"),
        ("                chunk_cu_num_logits_np,\n                max_num_logprobs,\n",
         "                chunk_cu_num_logits_np,\n                max_num_logprobs,\n"
         "                max_per_req_token_ids,\n"
         "                input_batch.expanded_idx_mapping[lo:hi],\n"),
        ("        chunk_logit_limit = get_max_chunk_logits(logits.shape[1])\n",
         "        max_per_req_token_ids = self.sampler.logprob_token_ids_state.max_num_token_ids(\n"
         "            input_batch.idx_mapping_np\n        )\n"
         "        chunk_logit_limit = get_max_chunk_logits(logits.shape[1])\n"),
        ("            chunk_logit_limit,\n            max_num_logprobs,\n",
         "            chunk_logit_limit,\n            max_num_logprobs,\n"
         "            max_per_req_token_ids,\n"),
    ],
    "v1/worker/gpu/model_runner.py": [
        ("                    # Rejection sampler does not return logprob token ids.\n"
         "                    include_token_ids=(\n"
         "                        global_input_batch.num_draft_tokens == 0\n"
         "                        or self.rejection_sampler is None\n                    ),\n",
         "                    # Both samplers preserve per-request candidate columns.\n"
         "                    include_token_ids=True,\n"),
    ],
    "entrypoints/launchers/app.py": [
        ("    init_exception_handler(app)\n",
         "    if 'generate' in supported_tasks:\n"
         "        from openjev.vllm_scores import install_routes\n"
         "        install_routes(app)\n\n    init_exception_handler(app)\n"),
    ],
}


def transform(name, source):
    for old, new in RULES[name]:
        if source.count(old) != 1:
            raise ValueError(f"{name}: expected one patch anchor, found {source.count(old)}")
        source = source.replace(old, new, 1)
    compile(source, name, "exec")
    return source


def prepare(root, *, revert=False):
    return prepare_sources(root, revert=revert,
                           manifest_name="b12x_logprobs_manifest.json", rules=RULES,
                           transform_source=transform,
                           backup_suffix=".forjev-logprobs-original")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
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
    print(f"B12X ForJev bridge: {len(prepared)} files {action}; restart required after writes")


if __name__ == "__main__":
    main()
