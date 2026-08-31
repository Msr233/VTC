import argparse
import json
from collections import defaultdict


def is_correct(value):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1"}
    return bool(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_file", default="", help="Path to the JSONL result file.")
    return parser.parse_args()


def update_counter(correct, total, group, is_correct):
    total[group] += 1
    if is_correct:
        correct[group] += 1


def print_report(title, correct, total):
    print("=" * 64)
    print(f"{title:^64}")
    print("=" * 64)
    print(f"{'Group':<28} | {'Samples':>7} | {'Correct':>7} | {'Accuracy':>8}")
    print("-" * 64)

    overall_total = 0
    overall_correct = 0
    for group in sorted(total):
        t = total[group]
        c = correct[group]
        acc = c / t * 100 if t else 0.0
        print(f"{str(group):<28} | {t:>7} | {c:>7} | {acc:>7.2f}%")
        overall_total += t
        overall_correct += c

    print("-" * 64)
    overall_acc = overall_correct / overall_total * 100 if overall_total else 0.0
    print(f"{'Overall':<28} | {overall_total:>7} | {overall_correct:>7} | {overall_acc:>7.2f}%")
    print("=" * 64)
    print(f"Overall accuracy: {overall_acc:.2f}% ({overall_correct}/{overall_total})")


def report(res):
    type_correct = defaultdict(int)
    type_total = defaultdict(int)
    qtype_correct = defaultdict(int)
    qtype_total = defaultdict(int)

    with open(res, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            correct_answer = is_correct(data.get("acc"))

            update_counter(
                type_correct,
                type_total,
                data.get("video_type") or data.get("type") or "unknown",
                correct_answer,
            )
            qtypes = data.get("question_type") or ["unknown"]
            if isinstance(qtypes, str):
                qtypes = [qtypes]
            for qtype in qtypes:
                update_counter(qtype_correct, qtype_total, qtype, correct_answer)

    print_report("LVBench Accuracy by Video Type", type_correct, type_total)
    print_report("LVBench Accuracy by Question Type", qtype_correct, qtype_total)


if __name__ == "__main__":
    args = parse_args()
    report(args.res_file)
