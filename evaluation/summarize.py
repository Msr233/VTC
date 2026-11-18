"""Aggregate completed question shards and report annotation coverage."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def question_ids(benchmark, annotations):
    if benchmark == "videomme":
        return {str(row["question_id"]) for row in annotations}
    if benchmark == "mlvu":
        return {str(row["qid"]) for row in annotations}
    if benchmark == "lvbench":
        return {f"{row['key']}_{row['uid']}" for row in annotations}
    return {str(row["id"]) for row in annotations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=("videomme", "mlvu", "lvbench", "longvideobench"),
        required=True,
    )
    parser.add_argument("--gt-file", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        nargs="+",
        type=Path,
        required=True,
        help="Prediction JSONL files (including files with .json extension)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    expected = question_ids(args.benchmark, json.loads(args.gt_file.read_text()))
    records = {}
    for path in args.predictions:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["id"])
            if qid in records:
                raise ValueError(f"Duplicate question ID: {qid}")
            if qid not in expected:
                raise ValueError(f"Question ID absent from annotation: {qid}")
            records[qid] = row
    correct = sum(str(row["acc"]).lower() == "true" for row in records.values())
    grouped = defaultdict(list)
    group_key = {
        "videomme": "duration_group",
        "mlvu": "category",
        "lvbench": "video_type",
        "longvideobench": "duration_group",
    }[args.benchmark]
    for row in records.values():
        grouped[str(row.get(group_key, "unspecified"))].append(
            str(row["acc"]).lower() == "true"
        )
    missing = sorted(expected - records.keys())
    result = {
        "benchmark": args.benchmark,
        "correct": correct,
        "expected_questions": len(expected),
        "evaluated_questions": len(records),
        "complete": not missing,
        "accuracy_percent": 100 * correct / len(expected) if expected else 0,
        "evaluated_accuracy_percent": 100 * correct / len(records) if records else 0,
        "missing_question_ids": missing,
        "groups": {
            key: {
                "correct": sum(values),
                "evaluated": len(values),
                "accuracy_percent": 100 * sum(values) / len(values),
            }
            for key, values in sorted(grouped.items())
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
