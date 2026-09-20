"""Create a self-contained dataset release and compressed archive; no upload."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/openjev-pilot/permissive-release")
    args = parser.parse_args()
    directory = args.dataset
    report = json.loads((directory / "report.json").read_text())
    audit = json.loads((directory / "tokenizer_audit.json").read_text())
    for name, info in report["files"].items():
        if checksum(directory / name) != info["sha256"]:
            raise ValueError(f"Dataset changed since build: {name}")
        split = name.split(".")[0]
        if audit["splits"][split]["records"] != info["records"]:
            raise ValueError("Tokenizer audit counts do not match")
        if audit["splits"][split]["jsonl_sha256"] != info["sha256"]:
            raise ValueError("Tokenizer audit is stale")
    parquet_dir = directory / "parquet"
    parquet_dir.mkdir(exist_ok=True)
    for split in ("train", "validation", "calibration"):
        flat = []
        with (directory / f"{split}.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                # Dynamic API dictionaries need strings to avoid huge Arrow structs.
                flat.append({k: row[k] for k in ("id", "state", "split", "group_id")} | {
                    k + "_json": json.dumps(row.get(k, {}), ensure_ascii=False, separators=(",", ":"))
                    for k in ("questions", "targets", "annotation_votes", "provenance")})
        table = pa.Table.from_pylist(flat)
        pq.write_table(table, parquet_dir / f"{split}.parquet", compression="zstd")
        restored = pq.read_table(parquet_dir / f"{split}.parquet").to_pylist()
        assert restored == flat, "Parquet roundtrip failed"
    for folder in ("dataset", "scripts", "openjev", "tests"):
        dest = directory / folder
        if folder == "dataset":
            shutil.copytree(ROOT / folder, dest, dirs_exist_ok=True)
        else:
            dest.mkdir(exist_ok=True)
            paths = ((ROOT / folder).glob("*.py") if folder == "openjev"
                     else [ROOT / folder / name for name in
                           (["build_public_dataset.py", "download_dataset_sources.py", "audit_public_dataset.py", "package_public_dataset.py"]
                            if folder == "scripts" else ["test_dataset.py"])])
            for path in paths:
                shutil.copy2(path, dest / path.name)
    for name in ("LICENSE", "pyproject.toml"):
        shutil.copy2(ROOT / name, directory / name)
    shutil.copy2(ROOT / "dataset/LICENSE-DATA.md", directory / "LICENSE-DATA.md")
    counts = {s: report["files"][s + ".jsonl"]["records"] for s in ("train", "validation", "calibration")}
    family_table = "\n".join(f'| {family} | {report["records_by_split_family"].get("train/" + family, 0):,} |'
                             for family in ("tasksource", "flan", "goemotions", "helpsteer"))
    card = '''---
language: [en]
license: other
license_name: mixed-permissive
license_link: LICENSE-DATA.md
task_categories: [text-classification]
size_categories: [10K<n<100K]
configs:
  - config_name: default
    data_files:
      - split: train
        path: parquet/train.parquet
      - split: validation
        path: parquet/validation.parquet
      - split: calibration
        path: parquet/calibration.parquet
---
# OpenJev public-data pilot

Permissively licensed public data compiled into OpenJev Choice, Noul, and Score
decisions. No Gemini-generated records or paid generation calls. This is a local
release candidate, not evidence of model quality or a completed model release.

'''
    card += f'**{sum(counts.values()):,} records:** {counts["train"]:,} train, {counts["validation"]:,} validation, {counts["calibration"]:,} calibration.\n\n'
    card += '| Training family | Records |\n|---|---:|\n' + family_table + '\n\n'
    card += '''The initial 20K FLAN target was not reached under strict licensing and
source-matching filters. Its retained subset is Cosmos QA only. The TaskSource
selection is NLI-heavy; HelpSteer is an additional ordinal supplement. No excluded
sources or duplicated rows were added to meet a numerical target.

Use the JSONL files directly with the OpenJev API schema. In the Parquet version,
`questions_json`, `targets_json`, `provenance_json`, and `annotation_votes_json`
are JSON strings so dynamic option dictionaries have a stable Arrow schema:

```python
import json
import pyarrow.parquet as pq

row = pq.read_table("parquet/train.parquet").slice(0, 1).to_pylist()[0]
request = {"state": row["state"], "questions": json.loads(row["questions_json"])}
targets = json.loads(row["targets_json"])
```

Targets and vote counts are supervision, never part of the model prompt.
GoEmotions votes are per-emotion Bernoulli counts, not a categorical emotion
distribution. Score uses original human integer ratings, not invented scores.

Read [the methodology](dataset/METHODOLOGY.md) for reproduction, split policy,
slot-loss requirements, calibration plans, and limitations. Read
[data licensing](LICENSE-DATA.md) and retain [source notices](dataset/notices/)
when redistributing. Records use Apache-2.0, BSD-3-Clause, or CC-BY-4.0; no blanket
relicensing is claimed. Build code is Apache-2.0.

`report.json` contains actual counts, rejection reasons, and hashes.
`tokenizer_audit.json` records validation against OpenJev's real tokenizer.
`near_duplicate_audit.json` records approximate cross-split duplicate removal.
`SHA256SUMS` covers every release file except itself. The original 3.13 GB source
cache is intentionally excluded from this release and can be reproduced using
the locked downloader. Training/model weights and inference evaluation are not
included because they have not been produced by this build.
'''
    (directory / "README.md").write_text(card)
    checksums = "".join(f"{checksum(path)}  {path.relative_to(directory)}\n"
                        for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "SHA256SUMS")
    (directory / "SHA256SUMS").write_text(checksums)
    archive = directory.parent / "openjev-public-pilot.tar.gz"
    # Stable ordering, zero timestamps, and fixed permissions for reproducibility.
    with archive.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w|") as tar:
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    info = tar.gettarinfo(str(path), arcname="openjev-public-pilot/" + str(path.relative_to(directory)))
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    with path.open("rb") as f:
                        tar.addfile(info, f)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{checksum(archive)}  {archive.name}\n")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": checksum(archive), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
