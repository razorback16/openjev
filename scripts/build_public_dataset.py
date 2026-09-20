"""Deterministic public-data compiler. See dataset/METHODOLOGY.md.

Run from the repository root with the `data` extra installed. Network is only
used by download_dataset_sources.py; this compiler runs entirely offline.
"""
from __future__ import annotations

import argparse
import collections
import csv
import fnmatch
import gzip
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from datasketch import MinHash, MinHashLSH

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260919


def normalized(s):
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def digest(*values):
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def wordform(s):
    return " ".join(re.findall(r"\w+", normalized(s)))


def rows(path):
    if path.suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16384):
            yield from batch.to_pylist()
    elif path.suffix in (".csv", ".tsv"):
        with path.open() as f:
            yield from csv.DictReader(f, delimiter="\t" if path.suffix == ".tsv" else ",")
    else:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as f:
            for line in f:
                yield json.loads(line)


def split_for(group):
    n = int(digest(SEED, "split", group)[:12], 16) / 16**12
    return "train" if n < .9 else "validation" if n < .95 else "calibration"


def excluded(task, evaluation):
    if task in evaluation.get("exact_pattern_exceptions", []):
        return False
    return any(fnmatch.fnmatchcase(task.lower(), p.lower()) for p in evaluation["excluded_task_patterns"])


def source_for(task, family, registry, evaluation):
    if excluded(task, evaluation):
        return None
    for name, source in registry["sources"].items():
        if (source["license"] in registry["allowed_licenses"] and source["family"] == family
                and any(fnmatch.fnmatchcase(task, p) for p in source["task_patterns"])):
            return name
    return None


def parse_tasksource(row):
    header, separator, state = row["inputs"].partition("\n")
    match = re.fullmatch(r'With no explanation, (label .+?) with either (.+)\.', header)
    if not separator or not match or not state.strip():
        raise ValueError("unsupported_tasksource_template")
    labels = re.findall(r'"([^"\n]+)"', match[2])
    if not 2 <= len(labels) <= 128 or len(set(labels)) != len(labels):
        raise ValueError("invalid_options")
    target = row["targets"]
    # TaskSource appends exactly one period, including after a label's punctuation.
    matching = [x for x in labels if target == x + "."]
    if len(matching) != 1:
        raise ValueError("unmatched_target")
    instruction = "Classify the relationship from text_A to text_B." if "text_A" in match[1] else "Classify the state."
    return state.strip(), labels, matching[0], instruction


class NativeIndex:
    """Verify underlying membership/labels and exclude held-out contexts."""
    def __init__(self):
        self.train = {}
        self.blocked = set()
        self.conflicting = set()

    def add(self, source, a, b, label, ref, train):
        key = digest(source, normalized(a), normalized(b))
        # Every question/variant sharing a premise stays in the same split.
        group = digest(source, normalized(a))
        if not train:
            self.blocked.add(group)
        elif key in self.train and self.train[key][0] != label:
            self.conflicting.add(key)
        else:
            self.train[key] = (label, ref, group)

    def get(self, source, a, b):
        key = digest(source, normalized(a), normalized(b))
        value = self.train.get(key)
        if key in self.conflicting or value is None or value[2] in self.blocked:
            return None
        return value


def native_tasksource(root):
    index = NativeIndex()
    for file in sorted((root / "native").glob("*/*")):
        name = file.parent.name
        if name not in ("babi_nli", "WANLI", "ruletaker", "logical-entailment") or "anonymized" in file.name:
            continue
        for i, row in enumerate(rows(file)):
            source = {"babi_nli": "babi", "WANLI": "wanli", "ruletaker": "ruletaker", "logical-entailment": "logical"}[name]
            if source == "babi":
                a, b, label = row["premise"], row["hypothesis"], ["not-entailed", "entailed"][row["label"]]
            elif source == "wanli":
                a, b, label = row["premise"], row["hypothesis"], row["gold"]
            elif source == "ruletaker":
                a, b, label = row["context"], row["question"], row["label"]
            else:
                a, b, label = row["A"], row["B"], row["label"]
            index.add(source, a, b, label, f"{file.relative_to(root)}:{i}", "train" in file.name)
    for split in ("train", "test"):
        file = root / f"banking_{split}.csv"
        for i, row in enumerate(rows(file)):
            index.add("banking77", row["text"], "", row["category"], f"{file.name}:{i}", split == "train")
    return index


def base_record(source, family, task, state, group, ref, registry, container_ref=None):
    entry = registry["sources"][source]
    return {"id": digest(source, normalized(state)), "state": state, "questions": {}, "targets": {},
            "split": split_for(group), "group_id": group,
            "provenance": {"family": family, "task": task, "source": source,
                           "upstream": entry["upstream"], "license": entry["license"],
                           "original_split": "train", "source_ref": ref, "container_ref": container_ref}}


def categorical(record, options, target, instruction, expansion_probability=.4, binary=False):
    rng = random.Random(digest(SEED, record["id"], "compile"))
    options = list(options)
    if binary:
        positive = next((x for x in options if x.lower() == "yes"), options[0])
        record["questions"]["q1"] = {"type": "noul", "instructions": instruction + f' Is "{positive}" the correct answer?'}
        record["targets"]["q1"] = "yes" if target == positive else "no"
        return record
    rng.shuffle(options)
    record["questions"]["q1"] = {"type": "choice", "instructions": instruction,
                                  "criteria": {x: x.replace("_", " ") for x in options}}
    record["targets"]["q1"] = target
    if rng.random() < expansion_probability:
        checks = [(target, "yes"), (rng.choice([x for x in options if x != target]), "no")]
        rng.shuffle(checks)  # q2 must not always be the positive target.
        for j, (label, answer) in enumerate(checks, 2):
            qid = f"q{j}"
            record["questions"][qid] = {"type": "noul", "instructions": instruction + f' Is "{label}" the correct answer?'}
            record["targets"][qid] = answer
    return record


def tasksource_candidates(root, registry, evaluation, rejects):
    index = native_tasksource(root)
    categories = json.loads((root / "banking_categories.json").read_text())
    allowed = {}
    for file in sorted(root.glob("tasksource_train*.parquet")):
        for i, row in enumerate(rows(file)):
            task = row["task"]
            if task not in allowed:
                allowed[task] = source_for(task, "tasksource", registry, evaluation)
            source = allowed[task]
            if source is None:
                rejects["tasksource/not_allowlisted_or_reserved"] += 1
                continue
            try:
                state, options, target, instruction = parse_tasksource(row)
                if source == "banking77":
                    a, b = state, ""
                    instruction = "Which banking issue does the customer describe?"
                else:
                    if not state.startswith("text_A: ") or "\ntext_B: " not in state:
                        raise ValueError("invalid_pair")
                    a, b = state[8:].split("\ntext_B: ", 1)
                    if source == "ruletaker":
                        a = a.removeprefix("What is not explicitly stated as true is considered false. \n")
                native = index.get(source, a, b)
                if native is None:
                    raise ValueError("not_original_train_or_heldout_context")
                if native[0] != target:
                    raise ValueError("native_label_mismatch")
                record = base_record(source, "tasksource", task, state, native[2], native[1], registry, f"{file.name}:{i}")
                if source == "banking77":
                    rng = random.Random(digest(SEED, record["id"], "cardinality"))
                    k = rng.choices([4, 8, 32, 77], [.1, .2, .2, .5])[0]
                    options = [target] + rng.sample([x for x in categories if x != target], k - 1)
                yield categorical(record, options, target, instruction, registry["expansion_probability"])
            except ValueError as e:
                rejects["tasksource/" + str(e)] += 1


class FlanMatcher:
    def __init__(self, root):
        self.lookup = collections.defaultdict(set)
        self.records = {}
        self.blocked = set()
        for source in ("cosmos",):
            for file in sorted(root.glob("cosmos_*")):
                if file.suffix not in (".csv", ".jsonl"):
                    continue
                for i, row in enumerate(rows(file)):
                    a, b = row["context"], row["question"]
                    label = int(row.get("label", -1))
                    group = digest(source, normalized(a))
                    if "train" not in file.name:
                        self.blocked.add(group)
                        continue
                    if label == -1:
                        continue
                    aw, bw = wordform(a), wordform(b)
                    key = digest(source, aw, bw)
                    if key in self.records and self.records[key][0] != label:
                        self.blocked.add(group)
                        continue
                    self.records[key] = (label, group, f"{file.name}:{i}", aw, bw,
                                         [row[f"answer{j}"] for j in range(4)])
                    anchor = max((aw, bw), key=len).split()
                    if len(anchor) >= 6:
                        self.lookup[(source, " ".join(anchor[:6]))].add(key)

    def match(self, source, prompt):
        text = wordform(prompt)
        words = text.split()
        candidates = set()
        for i in range(len(words) - 5):
            candidates.update(self.lookup.get((source, " ".join(words[i:i + 6])), ()))
        matches = [self.records[k] for k in candidates
                   if self.records[k][1] not in self.blocked
                   and self.records[k][3] in text and self.records[k][4] in text]
        # Reject ambiguous substring matches instead of guessing.
        return matches[0] if len(matches) == 1 else None


def flan_candidates(root, registry, evaluation, rejects):
    matcher = FlanMatcher(root)
    allowed = {}
    for file in sorted(root.glob("flan_train*.parquet")):
        for i, row in enumerate(rows(file)):
            task = row["task"]
            if task not in allowed:
                allowed[task] = source_for(task, "flan", registry, evaluation)
            source = allowed[task]
            if source is None:
                rejects["flan/not_allowlisted_or_reserved"] += 1
                continue
            try:
                if row["prompt"].count("\nOPTIONS:\n") != 1:
                    raise ValueError("not_single_option_block")
                state, opt_text = row["prompt"].rsplit("\nOPTIONS:\n", 1)
                lines = opt_text.splitlines()
                if not all(x.startswith("- ") for x in lines):
                    raise ValueError("unsupported_option_format")
                options = [x[2:] for x in lines]
                target = row["answer"]
                if len(set(options)) != len(options) or target not in options:
                    raise ValueError("invalid_options_or_target")
                native = matcher.match(source, state)
                if native is None:
                    raise ValueError("not_unique_original_train_or_heldout_context")
                if normalized(target) != normalized(native[5][native[0]]) or {normalized(x) for x in options} != {normalized(x) for x in native[5]}:
                    raise ValueError("native_label_or_options_mismatch")
                record = base_record(source, "flan", task, state.strip(), native[1], native[2], registry, f"{file.name}:{i}")
                # All templates of one native record share an ID, so keep one.
                record["id"] = digest(source, native[3], native[4])
                rename = lambda s: re.sub(r"(?i)none of the above choices\s*\.", "None of the other options.", s)
                yield categorical(record, [rename(x) for x in options], rename(target), "Answer the question in the state.",
                                  registry["expansion_probability"])
            except ValueError as e:
                rejects["flan/" + str(e)] += 1


def goemotions_candidates(root, registry, rejects):
    ids = {}
    held_text = set()
    for split in ("train", "dev", "test"):
        with (root / f"goemotions_{split}.tsv").open() as f:
            for text, _, rid in csv.reader(f, delimiter="\t"):
                ids[rid] = split
                if split != "train":
                    held_text.add(normalized(text))
    items = {}
    blocked_threads = set()
    emotions = None
    metadata = {"text", "id", "author", "subreddit", "link_id", "parent_id", "created_utc", "rater_id", "example_very_unclear"}
    for file in sorted(root.glob("goemotions_[123].csv")):
        for i, row in enumerate(rows(file)):
            if emotions is None:
                emotions = [x for x in row if x not in metadata]
            if ids.get(row["id"]) in ("dev", "test"):
                blocked_threads.add(row["link_id"])
            if ids.get(row["id"]) != "train" or normalized(row["text"]) in held_text:
                continue
            item = items.setdefault(row["id"], {"text": row["text"], "thread": row["link_id"], "votes": {}, "refs": []})
            if row["example_very_unclear"].lower() in ("true", "1"):
                continue
            item["votes"][row["rater_id"]] = {e: int(row[e]) for e in emotions}
            item["refs"].append(f"{file.name}:{i}")
    for rid, item in items.items():
        if item["thread"] in blocked_threads or len(item["votes"]) < 2:
            rejects["goemotions/heldout_thread_or_fewer_than_two_raters"] += 1
            continue
        group = digest("goemotions", item["thread"] or rid)
        rec = base_record("goemotions", "goemotions", "goemotions", item["text"], group, item["refs"], registry)
        rec["provenance"]["original_id"] = rid
        rng = random.Random(digest(SEED, rid))
        votes = list(item["votes"].values())
        counts = {e: sum(v[e] for v in votes) for e in emotions}
        active = [e for e in emotions if counts[e]]
        # Keep every emotion with any votes and one randomly selected negative.
        selected = active + rng.sample([e for e in emotions if not counts[e]], 1)
        rng.shuffle(selected)
        rater = rng.choice(votes)
        rec["annotation_votes"] = {}
        rec["provenance"]["target_policy"] = "one_seeded_rater_per_record; votes retained for evaluation only"
        for i, e in enumerate(selected, 1):
            qid = f"q{i}"
            rec["questions"][qid] = {"type": "noul", "instructions": f'Does this comment express {e}?' if e != "neutral" else "Is this comment emotionally neutral?"}
            rec["targets"][qid] = "yes" if rater[e] else "no"
            rec["annotation_votes"][qid] = {"yes": counts[e], "no": len(votes) - counts[e], "n_raters": len(votes)}
        yield rec


SCORE_CRITERIA = {
    "helpfulness": ["Not helpful", "Slightly helpful", "Somewhat helpful", "Mostly helpful", "Extremely helpful"],
    "correctness": ["Incorrect", "Mostly incorrect", "Partially correct", "Mostly correct", "Fully correct"],
    "coherence": ["Incoherent", "Mostly incoherent", "Somewhat coherent", "Mostly coherent", "Fully coherent"],
    "complexity": ["Basic", "Simple", "Intermediate", "Advanced", "Expert"],
    "verbosity": ["Succinct", "Pretty short", "Average length", "Moderately long", "Verbose"],
}


def helpsteer_candidates(root, registry, rejects):
    blocked = {normalized(r["prompt"]) for r in rows(root / "helpsteer_validation.jsonl.gz")}
    for i, row in enumerate(rows(root / "helpsteer_train.jsonl.gz")):
        if normalized(row["prompt"]) in blocked:
            rejects["helpsteer/heldout_prompt"] += 1
            continue
        state = json.dumps({"prompt": row["prompt"], "response": row["response"]}, ensure_ascii=False)
        rec = base_record("helpsteer", "helpsteer", "helpsteer", state,
                          digest("helpsteer", normalized(row["prompt"])), f"helpsteer_train.jsonl.gz:{i}", registry)
        for j, (attribute, criteria) in enumerate(SCORE_CRITERIA.items(), 1):
            value = row[attribute]
            if not isinstance(value, int) or not 0 <= value <= 4:
                raise ValueError("HelpSteer requires original integer ratings")
            qid = f"q{j}"
            rec["questions"][qid] = {"type": "score", "instructions": f"Rate the {attribute} of the response to the prompt.", "criteria": criteria}
            rec["targets"][qid] = value
        yield rec


def select_balanced(candidates, family, target, rejects):
    pools = collections.defaultdict(list)
    seen = set()
    for rec in candidates:
        if rec["id"] in seen:
            rejects[family + "/duplicate_original"] += 1
            continue
        seen.add(rec["id"])
        pools[(rec["split"], rec["provenance"]["task"])].append(rec)
    out = []
    for split, quota in (("train", target), ("validation", target // 20), ("calibration", target // 20)):
        buckets = []
        for (s, task), values in sorted(pools.items()):
            if s == split:
                values.sort(key=lambda r: digest(SEED, "select", r["id"]))
                buckets.append(iter(values))
        chosen = []
        while buckets and len(chosen) < quota:
            remaining = []
            for bucket in buckets:
                try:
                    chosen.append(next(bucket))
                    remaining.append(bucket)
                    if len(chosen) == quota:
                        break
                except StopIteration:
                    pass
            buckets = remaining
        print(f"{family}/{split}: {len(chosen)}/{quota}", flush=True)
        out.extend(chosen)
    return out


def validate(records, registry, evaluation):
    groups, states, ids = {}, {}, set()
    counts = collections.Counter()
    for rec in records:
        assert rec["id"] not in ids, "duplicate ID"
        ids.add(rec["id"])
        for key, mapping in ((rec["group_id"], groups), (normalized(rec["state"]), states)):
            assert key not in mapping or mapping[key] == rec["split"], "cross-split leakage"
            mapping[key] = rec["split"]
        assert set(rec["questions"]) == set(rec["targets"])
        p = rec["provenance"]
        assert source_for(p["task"], p["family"], registry, evaluation) == p["source"]
        assert p["original_split"] == "train"
        for qid, q in rec["questions"].items():
            target = rec["targets"][qid]
            if q["type"] == "choice":
                assert 2 <= len(q["criteria"]) <= 128 and target in q["criteria"]
            elif q["type"] == "score":
                assert 2 <= len(q["criteria"]) <= 10 and isinstance(target, int) and 0 <= target < len(q["criteria"])
            else:
                assert q["type"] == "noul" and target in ("yes", "no")
            counts[(rec["split"], q["type"])] += 1
    return {f"{s}/{k}": v for (s, k), v in sorted(counts.items())}


def near_duplicate_filter(records):
    """Conservative cross-split removal; approximate retrieval, exact similarity.

    Keep validation first, calibration next, training last. Removing a complete
    source group prevents a rejected variant surviving as another question.
    No claim of exhaustive semantic or benchmark decontamination is made.
    """
    lsh = MinHashLSH(threshold=.75, num_perm=64)
    retained = {}
    drop_groups = set()
    matches = []
    for split in ("validation", "calibration", "train"):
        pending = []
        for rec in sorted((r for r in records if r["split"] == split), key=lambda r: r["id"]):
            # Punctuation carries meaning in formal logic; do not erase it there.
            text = normalized(rec["state"])
            if rec["provenance"]["source"] == "helpsteer":
                text = normalized(json.loads(rec["state"])["prompt"])
            words = re.findall(r"\w+|[^\w\s]", text)
            shingles = {" ".join(words[i:i + 5]).encode() for i in range(max(1, len(words) - 4))}
            mh = MinHash(num_perm=64, seed=SEED)
            mh.update_batch(sorted(shingles))
            for candidate in sorted(lsh.query(mh)):
                other, other_shingles = retained[candidate]
                similarity = len(shingles & other_shingles) / len(shingles | other_shingles)
                if similarity >= .9:
                    drop_groups.add(rec["group_id"])
                    matches.append({"removed_id": rec["id"], "retained_id": other["id"], "jaccard": similarity})
                    break
            if split != "train":
                pending.append((rec, shingles, mh))
        for rec, shingles, mh in pending:
            if rec["group_id"] not in drop_groups:
                lsh.insert(rec["id"], mh)
                retained[rec["id"]] = (rec, shingles)
    kept = [r for r in records if r["group_id"] not in drop_groups]
    return kept, {"method": "64-permutation MinHash LSH (retrieval threshold .75), exact 5-token-shingle Jaccard >= .90",
                  "scope": "across compiled splits; not external benchmarks or exhaustive semantic matching",
                  "removed_groups": len(drop_groups), "removed_records": len(records) - len(kept),
                  "matched_pairs": matches}


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    global SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/openjev-pilot/source")
    parser.add_argument("--output", type=Path, default=ROOT / "data/openjev-pilot/permissive-release")
    args = parser.parse_args()
    registry = yaml.safe_load((ROOT / "dataset/training_sources.yaml").read_text())
    SEED = registry["seed"]
    if registry["split_proportions"] != {"train": .9, "validation": .05, "calibration": .05}:
        raise ValueError("This compiler currently supports only the documented 90/5/5 split")
    evaluation = yaml.safe_load((ROOT / "dataset/evaluation_sources.yaml").read_text())
    quality = yaml.safe_load((ROOT / "dataset/quality_exclusions.yaml").read_text())["original_refs"]
    rejects = collections.Counter()
    records = []
    generators = {
        "tasksource": lambda: tasksource_candidates(args.source, registry, evaluation, rejects),
        "flan": lambda: flan_candidates(args.source, registry, evaluation, rejects),
        "goemotions": lambda: goemotions_candidates(args.source, registry, rejects),
        "helpsteer": lambda: helpsteer_candidates(args.source, registry, rejects),
    }
    for family, generate in generators.items():
        records.extend(select_balanced(generate(), family, registry["train_targets"][family], rejects))
    # Exact cross-source state duplicates: keep one, never repair splits after expansion.
    unique = {}
    for rec in sorted(records, key=lambda r: r["id"]):
        refs = rec["provenance"]["source_ref"]
        if any(ref in quality for ref in (refs if isinstance(refs, list) else [refs])):
            rejects["global/sampled_quality_exclusion"] += 1
            continue
        key = normalized(rec["state"])
        if key in unique:
            rejects["global/duplicate_state"] += 1
        else:
            unique[key] = rec
    records = list(unique.values())
    print("Auditing cross-split near duplicates...", flush=True)
    records, near_audit = near_duplicate_filter(records)
    slots = validate(records, registry, evaluation)
    args.output.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in ("train", "validation", "calibration"):
        selected = sorted((r for r in records if r["split"] == split), key=lambda r: digest(SEED, "order", r["id"]))
        path = args.output / (split + ".jsonl")
        with path.open("w") as f:
            for rec in selected:
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        files[path.name] = {"records": len(selected), "bytes": path.stat().st_size,
                            "sha256": file_sha256(path)}
    report = {"schema_version": 1, "seed": SEED, "gemini_records": 0, "files": files,
              "requested_train_counts": registry["train_targets"],
              "records_by_split_family": dict(sorted(collections.Counter(f'{r["split"]}/{r["provenance"]["family"]}' for r in records).items())),
              "records_by_split_task": dict(sorted(collections.Counter(f'{r["split"]}/{r["provenance"]["task"]}' for r in records).items())),
              "slots": slots, "rejections": dict(sorted(rejects.items())),
              "choice_cardinalities": dict(sorted(collections.Counter(str(len(q["criteria"])) for r in records for q in r["questions"].values() if q["type"] == "choice").items())),
              "validation": {"duplicate_ids": 0, "cross_split_groups": 0, "cross_split_exact_states": 0,
                             "original_train_membership_checked": True, "reserved_families": 0,
                             "external_benchmark_content_scan": "not performed; family exclusions only",
                             "near_duplicate_scan": {k: v for k, v in near_audit.items() if k != "matched_pairs"}}}
    (args.output / "near_duplicate_audit.json").write_text(json.dumps(near_audit, indent=2) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"files": files, "slots": slots}, indent=2), flush=True)


if __name__ == "__main__":
    main()
