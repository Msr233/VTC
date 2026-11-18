#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
IFS=',' read -r -a gpus <<< "${GPU_GROUP:-0}"
count="${#gpus[@]}"
root="${OUTPUT_ROOT:-outputs/suite}"
for benchmark in ${BENCHMARKS:-videomme mlvu lvbench longvideobench}; do
    prefix="${benchmark^^}"
    video_key="${prefix}_VIDEO_DIR"
    gt_key="${prefix}_GT_FILE"
    videos="${!video_key:?Set the benchmark VIDEO_DIR variable}"
    annotation="${!gt_key:?Set the benchmark GT_FILE variable}"
    for method in ${METHODS:-vtc vtcdy}; do
        ratios="${RATIOS:-0.5 0.25}"
        if [[ "$method" == none ]]; then ratios="1.0"; fi
        for ratio in $ratios; do
            out="$root/$benchmark/${method}_r${ratio}"
            pids=()
            predictions=()
            for ((k=0; k<count; k++)); do
                shard="$out/shard${k}"
                predictions+=("$shard/${count}_${k}.json")
                CUDA_VISIBLE_DEVICES="${gpus[k]}" VIDEO_DIR="$videos" GT_FILE="$annotation" \
                OUTPUT_DIR="$shard" bash scripts/run_benchmark.sh "$benchmark" "$method" "$ratio" \
                    --num-chunks "$count" --chunk-idx "$k" --output-name "${count}_${k}" "$@" &
                pids+=("$!")
            done
            failed=0
            for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
            if ((failed)); then
                echo "Evaluation failed: $benchmark $method $ratio; see shard logs" >&2
                exit 1
            fi
            "${PYTHON:-python}" -m evaluation.summarize --benchmark "$benchmark" \
                --gt-file "$annotation" --predictions "${predictions[@]}" --output "$out/metrics.json"
        done
    done
done
