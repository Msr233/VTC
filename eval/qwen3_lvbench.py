import argparse
import json
import math
import os
import random
import re

import numpy as np
import torch
from qwen_vl_utils_new_2 import process_mosaic_video_vision_info, process_vision_info
from vtc_token_stats import latest_vtc_token_stats
from vtc_context import clear_vtc_video_contexts, set_vtc_video_contexts
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from attention_backend import resolve_attention_backend


def split_list(lst, n):
    """Split a list into n roughly equal-sized chunks."""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return lst[k::n]


LVBENCH_POST_PROMPT = "\nAnswer the question with the option letter"

_TRANSFORMERS_VIDEO_METADATA_KEYS = {
    "total_num_frames",
    "fps",
    "width",
    "height",
    "duration",
    "video_backend",
    "frames_indices",
}


def _processor_video_metadata(video_metadatas):
    if video_metadatas is None:
        return None
    return [
        {
            key: value
            for key, value in dict(metadata or {}).items()
            if key in _TRANSFORMERS_VIDEO_METADATA_KEYS
        }
        for metadata in video_metadatas
    ]


def build_lvbench_prompt(question):
    return question.rstrip() + LVBENCH_POST_PROMPT


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index, e.g. A, B, C, D.
    """
    response = response.strip()
    for char in [",", ".", "!", "?", ";", ":", "'", '"', ")"]:
        response = response.strip(char)
    padded_response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in padded_response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for match in re.finditer(r"(?<![A-Z])([A-Z])(?=[\)\.\s,:;]|$)", response):
            choice = match.group(1)
            if choice in all_choices:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(padded_response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(padded_response.rfind(can))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        pred_index = candidates[int(np.argmax(start_indexes))]
    else:
        pred_index = candidates[0]

    return pred_index


def parse_options_from_question(question):
    option_dict = re.findall(
        r"\(([A-Z])\)\s*([^\n]+?)(?=\n\([A-Z]\)|$)",
        question.strip(),
        flags=re.S,
    )
    return [(idx, answer.strip()) for idx, answer in option_dict]


def build_questions(gt_file):
    data = json.load(open(gt_file, "r", encoding="utf-8"))
    gt_questions = []
    for row in data:
        option_dict = parse_options_from_question(row["question"])
        index2ans = {idx: ans for idx, ans in option_dict}
        answer_id = row["answer"]
        answer = index2ans.get(answer_id, answer_id)
        video_name = row.get("video_path") or f"{row['key']}.mp4"
        video_id = os.path.splitext(os.path.basename(video_name))[0]

        gt_questions.append(
            {
                "qid": f"{row['key']}_{row['uid']}",
                "question": row["question"],
                "video_name": video_name,
                "video_id": video_id,
                "answer_id": answer_id,
                "answer": answer,
                "index2ans": index2ans,
                "question_type": row.get("question_type", []),
                "video_type": row.get("type", ""),
                "time_reference": row.get("time_reference", ""),
            }
        )
    return gt_questions


def run_inference(args):
    args.attn_implementation = resolve_attention_backend(args.attn_implementation)
    if args.multi_gpu:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            device_map="auto",
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
        )
        model.to("cuda:0")

    if args.enable_vtc:
        from models import apply_vtc_dy

        pruner_kwargs = dict(
            retention_ratio=args.vtc_retention_ratio,
            semantic_i_threshold=args.vtc_semantic_i_threshold,
            p_frame_min_tokens=args.vtc_p_frame_min_tokens,
            fill_mode=args.semantic_fill_mode,
            window_size=args.semantic_window_size,
            warmup_frames=args.semantic_warmup_frames,
            enable_vit_chunking=args.enable_vit_chunking,
            vit_chunk_frames=args.vit_chunk_frames,
            vit_chunk_staging_device=args.vit_chunk_staging_device,
            vit_chunk_collect_stats=args.vit_chunk_collect_stats,
            vit_chunk_verify_equivalence=args.vit_chunk_verify_equivalence,
        )
        pruner_kwargs.update(
            codec_bitcost_enabled=args.semantic_codec_bitcost_enabled,
            token_allocation_mode=args.vtc_token_allocation_mode,
            codec_cache_dir=args.semantic_codec_cache_dir,
            codec_ffprobe_path=args.semantic_codec_ffprobe_path,
            codec_strict=args.semantic_codec_strict,
            codec_aggregation=args.semantic_codec_aggregation,
            codec_mean_weight=args.semantic_codec_mean_weight,
            codec_p90_weight=args.semantic_codec_p90_weight,
            codec_normalization=args.semantic_codec_normalization,
            codec_i_policy=args.semantic_codec_i_policy,
            codec_z_clip=args.semantic_codec_z_clip,
            codec_budget_power=args.semantic_codec_budget_power,
            codec_budget_max_multiplier=args.semantic_codec_budget_max_multiplier,
            iframe_compression_mode=args.vtc_i_frame_compression,
            i_frame_retention_ratio=args.vtc_i_frame_retention_ratio,
            first_iframe_keep_ratio=args.semantic_first_iframe_keep_ratio,
            dynamic_iframe_keep_ratio=args.semantic_dynamic_iframe_keep_ratio,
            iframe_min_keep_ratio=args.semantic_iframe_min_keep_ratio,
            iframe_anchor_ratio=args.semantic_iframe_anchor_ratio,
            iframe_selection_mode=args.semantic_iframe_selection_mode,
            iframe_anchor_grid=args.semantic_iframe_anchor_grid,
            i_frame_residual_weight=args.vtc_i_frame_residual_weight,
            iframe_compress_when_ratio_one=args.semantic_iframe_compress_when_ratio_one,
            token_selection_mode=args.vtc_token_selection_mode,
            region_grid_h=args.vtc_region_grid_h,
            region_grid_w=args.vtc_region_grid_w,
            coverage_ratio=args.vtc_coverage_ratio,
            region_min_tokens=args.vtc_region_min_tokens,
        )
        apply_vtc_dy(
            model,
            version=args.semantic_version,
            **pruner_kwargs,
        )

    processor = AutoProcessor.from_pretrained(args.model_path)

    gt_questions = build_questions(args.gt_file)
    random.seed(42)
    random.shuffle(gt_questions)

    questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    os.makedirs(args.output_dir, exist_ok=True)
    output_name = f"{args.num_chunks}_{args.chunk_idx}" if args.num_chunks > 1 else args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")

    existing_ids = set()
    if os.path.exists(answers_file):
        with open(answers_file, "r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "id" in rec:
                    existing_ids.add(rec["id"])

    with open(answers_file, "a", encoding="utf-8") as ans_file:
        for line in tqdm(questions):
            qid = line["qid"]
            if qid in existing_ids:
                continue

            video_id = line["video_id"]
            question = build_lvbench_prompt(line["question"])
            sample_set = {
                "id": qid,
                "video_id": video_id,
                "question": question,
                "answer_id": line["answer_id"],
                "answer": line["answer"],
                "question_type": line["question_type"],
                "video_type": line["video_type"],
                "time_reference": line["time_reference"],
            }
            video_path = os.path.join(args.video_dir, line["video_name"])

            if not os.path.exists(video_path):
                if args.debug:
                    print(f"Miss video {video_path}")
                continue

            video_content = {
                "type": "video",
                "video": video_path,
                "nframes": args.topk,
                "semantic_version": args.semantic_version,
                "question_id": qid,
                "sample_id": qid,
            }
            if args.max_pixels and args.max_pixels > 0:
                video_content["max_pixels"] = args.max_pixels

            messages = [
                {
                    "role": "user",
                    "content": [
                        video_content,
                        {"type": "text", "text": question},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if args.video_load_branch == "mosaic":
                images, videos, video_kwargs = process_mosaic_video_vision_info(
                    messages,
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                    mosaic_source_frames=args.mosaic_source_frames,
                    read_batch_size=args.mosaic_read_batch_size,
                    frame_extract_version=args.frame_extract_version,
                    mosaic_packing_order=args.mosaic_packing_order,
                    mosaic_pair_pack_order=args.mosaic_pair_pack_order,
                    mosaic_require_homogeneous_pair=args.mosaic_require_homogeneous_pair,
                    mosaic_strict_pairing=args.mosaic_strict_pairing,
                    mosaic_dump_pair_mapping=args.mosaic_dump_pair_mapping,
                    dyrso_i_pair_mode=args.dyrso_i_pair_mode,
                    dyrso_hevc_pair_max_weight=args.dyrso_hevc_pair_max_weight,
                    dyrso_i_pair_candidates=args.dyrso_i_pair_candidates,
                    dyrso_i_pair_strict=args.dyrso_i_pair_strict,
                    dyrso_dump_i_pair_selection=args.dyrso_dump_i_pair_selection,
                    dyrso_codec_cache_dir=args.semantic_codec_cache_dir,
                    dyrso_codec_ffprobe_path=args.semantic_codec_ffprobe_path,
                    dyrso_codec_z_clip=args.semantic_codec_z_clip,
                )
            else:
                images, videos, video_kwargs = process_vision_info(
                    messages,
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
                if args.debug:
                    for i, v in enumerate(videos):
                        print(
                            f"[Video {i}] raw_frames={v.shape[0]} shape={list(v.shape)} "
                            f"fps={video_metadatas[i].get('fps', '?')} "
                            f"total_frames={video_metadatas[i].get('total_num_frames', '?')}"
                        )
            else:
                video_metadatas = None

            inputs = processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=_processor_video_metadata(video_metadatas),
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            )
            inputs = inputs.to(model.device)

            generation_kwargs = {
                "do_sample": False,
                "top_p": float(os.getenv("top_p", 1.0)),
                "top_k": int(os.getenv("top_k", 20)),
                "temperature": float(os.getenv("temperature", 1.0)),
                "repetition_penalty": float(os.getenv("repetition_penalty", 1.0)),
                "max_new_tokens": int(os.getenv("out_seq_length", 16)),
            }

            semantic_log_context = {
                "sample_id": qid,
                "question_id": qid,
                "video_id": video_id,
            }
            inner_model = getattr(model, "model", None)
            vtc = getattr(inner_model, "_vtc", None)
            for target in (model, inner_model, vtc):
                if target is not None:
                    setattr(target, "_semantic_log_context", semantic_log_context)

            try:
                set_vtc_video_contexts(model, [video_path], video_metadatas, inputs)
                generated_ids = model.generate(**inputs, **generation_kwargs)
            finally:
                clear_vtc_video_contexts(model)
                for target in (model, inner_model, vtc):
                    if target is not None and hasattr(target, "_semantic_log_context"):
                        delattr(target, "_semantic_log_context")
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            outputs = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            parsed_pred = parse_multi_choice_response(
                outputs,
                ["A", "B", "C", "D"],
                line["index2ans"],
            )
            sample_set["acc"] = str(parsed_pred == line["answer_id"])
            sample_set["parsed_pred"] = parsed_pred
            sample_set["pred"] = outputs
            vtc_token_stats = latest_vtc_token_stats(
                model,
                fallback_video_id=video_id,
            )
            if vtc_token_stats is not None:
                sample_set["vtc_token_stats"] = vtc_token_stats
            ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
            ans_file.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", help="Directory containing video files.", required=True)
    parser.add_argument("--gt_file", help="Path to the ground truth file.", required=True)
    parser.add_argument("--output_dir", help="Directory to save result JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the output JSON file.", required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--attn_implementation", type=str, default="flash_attention_2"
    )

    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        default=False,
        help="Use device_map='auto' to split the model across visible GPUs.",
    )
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=0,
        help="Optional per-frame video max_pixels. 0 keeps qwen-vl-utils dynamic default.",
    )
    parser.add_argument("--video_load_branch", type=str, default="default", choices=["default", "mosaic"])
    parser.add_argument("--mosaic_source_frames", type=int, default=8)
    parser.add_argument("--mosaic_read_batch_size", type=int, default=256)
    parser.add_argument("--frame_extract_version", type=str, default="v1", choices=["v0", "v1"])
    parser.add_argument("--mosaic_packing_order", type=str, default=None, choices=["qwen3_2i2p", "miniset_1i1p"])
    parser.add_argument("--mosaic_pair_pack_order", type=str, default="sequential", choices=["sequential", "pair_aligned"])
    parser.add_argument("--mosaic_require_homogeneous_pair", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mosaic_strict_pairing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mosaic_dump_pair_mapping", action="store_true", default=False)
    parser.add_argument("--dyrso_i_pair_mode", type=str, default="fixed", choices=["fixed", "hevc_bitcost", "hevc_tile_sort"])
    parser.add_argument("--dyrso_hevc_pair_max_weight", type=float, default=0.3)
    parser.add_argument("--dyrso_i_pair_candidates", type=str, default=None)
    parser.add_argument("--dyrso_i_pair_strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dyrso_dump_i_pair_selection", action="store_true", default=False)

    parser.add_argument("--enable_vtc", action="store_true", default=False)
    parser.add_argument("--semantic_version", type=str, default="v5.0.0", choices=["v5.0.0", "v500"])
    parser.add_argument("--vtc_retention_ratio", type=float, default=0.5)
    parser.add_argument("--vtc_semantic_i_threshold", type=float, default=2.0)
    parser.add_argument("--vtc_p_frame_min_tokens", type=int, default=8)
    parser.add_argument("--semantic_max_iframes", type=int, default=-1)
    parser.add_argument("--semantic_min_iframe_gap", type=int, default=-1)
    parser.add_argument("--semantic_budget_infeasible_policy", type=str, default="adjust_p_min", choices=["adjust_p_min", "error"])
    parser.add_argument("--semantic_enable_iframe_prefix_quota", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vtc_token_allocation_mode", type=str, default="dyta", choices=["uniform", "residual_mean", "dyta"])
    parser.add_argument("--semantic_codec_bitcost_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--semantic_codec_cache_dir",
        type=str,
        default="./outputs_codec_cache_v2",
    )
    parser.add_argument("--semantic_codec_ffprobe_path", type=str, default="ffprobe")
    parser.add_argument("--semantic_codec_strict", action="store_true", default=False)
    parser.add_argument("--semantic_codec_aggregation", type=str, default="mean_p90", choices=["mean", "p90", "mean_p90", "max", "sum"])
    parser.add_argument("--semantic_codec_mean_weight", type=float, default=0.5)
    parser.add_argument("--semantic_codec_p90_weight", type=float, default=0.5)
    parser.add_argument("--semantic_codec_normalization", type=str, default="robust_pb", choices=["raw_bits", "bpp", "log_bpp", "robust_pb"])
    parser.add_argument("--semantic_codec_i_policy", type=str, default="exclude", choices=["exclude", "nearest_pb", "clip_as_pb"])
    parser.add_argument("--semantic_codec_z_clip", type=float, default=5.0)
    parser.add_argument("--semantic_codec_budget_power", type=float, default=0.5)
    parser.add_argument("--semantic_codec_budget_max_multiplier", type=float, default=2.0)
    parser.add_argument("--vtc_i_frame_compression", type=str, default="lightweight", choices=["none", "lightweight"])
    parser.add_argument("--vtc_i_frame_retention_ratio", type=float, default=0.9)
    parser.add_argument("--semantic_first_iframe_keep_ratio", type=float, default=None)
    parser.add_argument("--semantic_dynamic_iframe_keep_ratio", type=float, default=None)
    parser.add_argument("--semantic_iframe_min_keep_ratio", type=float, default=0.8)
    parser.add_argument("--semantic_iframe_anchor_ratio", type=float, default=0.2)
    parser.add_argument("--semantic_iframe_selection_mode", type=str, default="spatial_anchor_salient", choices=["spatial_anchor_salient", "salient_only", "random"])
    parser.add_argument("--semantic_iframe_anchor_grid", type=str, default="4x6")
    parser.add_argument("--vtc_i_frame_residual_weight", type=float, default=0.5)
    parser.add_argument("--semantic_iframe_compress_when_ratio_one", action="store_true", default=False)
    parser.add_argument(
        "--semantic_fill_mode",
        type=str,
        default="zero",
        choices=["zero", "mean", "previous"],
    )
    parser.add_argument("--vtc_token_selection_mode", type=str, default="sc_topk",
                        choices=["sc_topk", "global_topk"])
    parser.add_argument("--vtc_region_grid_h", type=int, default=4)
    parser.add_argument("--vtc_region_grid_w", type=int, default=4)
    parser.add_argument("--vtc_coverage_ratio", type=float, default=0.20)
    parser.add_argument("--vtc_region_min_tokens", type=int, default=1)
    parser.add_argument("--semantic_window_size", type=int, default=8)
    parser.add_argument("--semantic_warmup_frames", type=int, default=2)
    parser.add_argument("--enable_vit_chunking", action="store_true", default=False)
    parser.add_argument("--vit_chunk_frames", type=int, default=32)
    parser.add_argument("--vit_chunk_staging_device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--vit_chunk_collect_stats", action="store_true", default=False)
    parser.add_argument("--vit_chunk_verify_equivalence", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug-level log output (default: progress bar only).")

    args = parser.parse_args()
    args.video_load_branch = "default" if args.frame_extract_version == "v0" else "mosaic"
    return args


if __name__ == "__main__":
    run_inference(parse_args())
