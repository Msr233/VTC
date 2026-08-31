import argparse
import json
from collections import defaultdict

try:
    from .vtc_token_stats import print_vtc_token_report
except ImportError:
    from vtc_token_stats import print_vtc_token_report


def is_correct(value):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1"}
    return bool(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_file", required=True, help="Path to result jsonl file.")
    return parser.parse_args()


def report(res_file):
    correct = defaultdict(int)
    total = defaultdict(int)
    records = []

    with open(res_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(data)
            group = data.get("duration_group", data.get("duration", "unknown"))
            total[group] += 1
            if is_correct(data.get("acc")):
                correct[group] += 1

    print("=" * 52)
    print(f"{'Accuracy Report':^52}")
    print("=" * 52)
    print(f"{'Duration Group':<15} | {'Samples':>7} | {'Correct':>7} | {'Accuracy':>8}")
    print("-" * 52)

    overall_total = 0
    overall_correct = 0
    for group in sorted(total):
        t = total[group]
        c = correct[group]
        acc = c / t * 100
        print(f"{str(group):<15} | {t:>7} | {c:>7} | {acc:>7.2f}%")
        overall_total += t
        overall_correct += c

    print("-" * 52)
    overall_acc = overall_correct / overall_total * 100 if overall_total else 0.0
    print(f"{'Overall':<15} | {overall_total:>7} | {overall_correct:>7} | {overall_acc:>7.2f}%")
    print("=" * 52)
    print_vtc_token_report(records)


if __name__ == "__main__":
    report(parse_args().res_file)
