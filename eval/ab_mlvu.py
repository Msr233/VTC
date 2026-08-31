import argparse
import json
from collections import defaultdict


SUPER_CATEGORIES = {
    "Holistic": ["7_topic_reasoning", "6_anomaly_reco"],
    "Multi_Detail": ["4_count", "5_order"],
    "Single_Detail": ["3_ego", "2_needle", "1_plotQA"],
}


def is_correct(value):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1"}
    return bool(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_file", default="", help="Path to the JSONL result file.")
    return parser.parse_args()


def report(res):
    correct = defaultdict(int)
    total = defaultdict(int)

    with open(res, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            group = data.get("category") or data.get("video_id", "unknown").split("/")[0]
            total[group] += 1
            if is_correct(data.get("acc")):
                correct[group] += 1

    print("=" * 64)
    print(f"{'MLVU Accuracy Report':^64}")
    print("=" * 64)
    print(f"{'Category':<20} | {'Samples':>7} | {'Correct':>7} | {'Accuracy':>8}")
    print("-" * 64)

    overall_total = 0
    overall_correct = 0
    for group in sorted(total):
        t = total[group]
        c = correct[group]
        acc = c / t * 100 if t else 0.0
        print(f"{str(group):<20} | {t:>7} | {c:>7} | {acc:>7.2f}%")
        overall_total += t
        overall_correct += c

    print("=" * 64)
    print(f"{'Super Class Statistics':^64}")
    print("-" * 64)
    for category_name, sub_types in SUPER_CATEGORIES.items():
        cat_total = sum(total[t] for t in sub_types)
        cat_correct = sum(correct[t] for t in sub_types)
        cat_acc = cat_correct / cat_total * 100 if cat_total else 0.0
        print(f"{category_name:<20} | {cat_total:>7} | {cat_correct:>7} | {cat_acc:>7.2f}%")

    print("-" * 64)
    overall_acc = overall_correct / overall_total * 100 if overall_total else 0.0
    print(f"{'Overall':<20} | {overall_total:>7} | {overall_correct:>7} | {overall_acc:>7.2f}%")
    print("=" * 64)
    print(f"Overall accuracy: {overall_acc:.2f}% ({overall_correct}/{overall_total})")


if __name__ == "__main__":
    args = parse_args()
    report(args.res_file)
