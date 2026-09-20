"""Audit compiled JSONL against OpenJev's real tokenizer and slot templates."""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openjev.config import Settings
from openjev.engine import Engine
from transformers import AutoTokenizer


async def audit(directory, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    engine = Engine(Settings(canvas=64), tokenizer)
    shapes = set()
    report = {"tokenizer": tokenizer_name, "tokenizer_json_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
              "available_choice_labels": len(engine.choice_labels), "model_inference_performed": False,
              "versions": {p: importlib.metadata.version(p) for p in ("transformers", "tokenizers", "pyarrow", "datasketch", "PyYAML")},
              "splits": {}}
    try:
        for split in ("train", "validation", "calibration"):
            sizes = []
            file_hash = hashlib.sha256()
            counts = collections.Counter()
            target_positions = collections.Counter()
            with (directory / (split + ".jsonl")).open() as f:
                for i, line in enumerate(f):
                    file_hash.update(line.encode("utf-8"))
                    record = json.loads(line)
                    schema = engine.build_schema(record["questions"])
                    qs, fmt = schema["questions"], schema["format"]
                    signature = tuple((q["id"], q["type"], tuple(q["labels"])) for q in qs)
                    if signature not in shapes:
                        for group in engine.groups(qs, fmt):
                            _, slots = engine.resolve_template(group, fmt)
                            assert len({t for s in slots for t in s["label_ids"]}) <= 128
                        shapes.add(signature)
                    for q in qs:
                        target = record["targets"][q["key"]]
                        values = [name for name, _ in q["choices"]]
                        position = values.index(str(target))
                        counts[q["type"]] += 1
                        target_positions[f'{q["type"]}/{len(values)}/{position}'] += 1
                    system = engine.system_text(qs, fmt)
                    length = len(engine.chat_prompt_ids(system, record["state"]))
                    if length + 64 > 65536:
                        raise ValueError(f'Context limit exceeded: {record["id"]}')
                    sizes.append(length)
                    if (i + 1) % 20000 == 0:
                        print(f"Audited {split}: {i + 1}", flush=True)
            sizes.sort()
            report["splits"][split] = {"records": len(sizes), "jsonl_sha256": file_hash.hexdigest(), "slots": dict(counts), "target_positions": dict(sorted(target_positions.items())),
                                      "prompt_tokens": {"min": sizes[0], "median": sizes[len(sizes) // 2],
                                                        "p95": sizes[int(len(sizes) * .95)], "max": sizes[-1], "sum": sum(sizes)}}
        report["unique_template_shapes"] = len(shapes)
        report["invalid_templates"] = 0
        report["invalid_targets"] = 0
        report["over_context_limit"] = 0
        (directory / "tokenizer_audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "splits"}, indent=2), flush=True)
    finally:
        await engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/openjev-pilot/permissive-release")
    parser.add_argument("--tokenizer", default="nvidia/diffusiongemma-26B-A4B-it-NVFP4")
    args = parser.parse_args()
    asyncio.run(audit(args.dataset, args.tokenizer))


if __name__ == "__main__":
    main()
