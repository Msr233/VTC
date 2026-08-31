from transformers import AutoModelForImageTextToText, AutoProcessor
from attention_backend import resolve_attention_backend
from qwen_vl_utils_new_2 import process_mosaic_video_vision_info, process_vision_info
from vtc_context import clear_vtc_video_contexts, set_vtc_video_contexts
import os
import torch
import argparse
import math
import numpy as np

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    return lst[k::n]


LONGVIDEOBENCH_POST_PROMPT_LMMS = "Answer with the option's letter from the given choices directly.\n"
LONGVIDEOBENCH_POST_PROMPT = "Please answer directly with only the letter of the correct option and nothing else.\n"


def build_longvideobench_prompt(question, options):
    return question + "\n" + "\n".join(options) + "\n" + LONGVIDEOBENCH_POST_PROMPT


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index

def get_option_prompt(candidates, version="default"):
    option_prompt = ""
    options = []
    for idx, candidate in enumerate(candidates):
        choice = chr(ord("A") + idx)
        if version == "v4":
            option_prompt += f"({choice}) {candidate}\n"
        else:
            option_prompt += f"({choice}):{candidate} "
        options.append(choice)
    options = "(" + ",".join(options) + ")"
    return option_prompt, options  

import json
import random
from tqdm import tqdm
from vtc_token_stats import latest_vtc_token_stats


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


def _empty_mosaic_pair_totals():
    return {
        "total_mosaic_pairs": 0,
        "total_tile_pairs": 0,
        "misaligned_tile_pairs": 0,
        "heterogeneous_temporal_pairs": 0,
        "pair_aligned_fallbacks": 0,
    }


def _empty_dyrso_i_pair_totals():
    return {
        "selected_pair_histogram": {},
        "output_pattern_counts": {"IIPP": 0},
        "alignment_failure_count": 0,
        "fallback_count": 0,
        "invalid_candidate_count": 0,
        "heterogeneous_pair_count": 0,
    }


def _accumulate_mosaic_pair_totals(totals, video_metadatas):
    if not video_metadatas:
        return
    for metadata in video_metadatas:
        stats = dict((metadata or {}).get("mosaic_pair_stats") or {})
        for key in totals:
            totals[key] += int(stats.get(key, 0) or 0)


def _accumulate_dyrso_i_pair_totals(totals, video_metadatas):
    if not video_metadatas:
        return
    for metadata in video_metadatas:
        stats = dict((metadata or {}).get("dyrso_i_pair_stats") or {})
        for key in (
            "alignment_failure_count",
            "fallback_count",
            "invalid_candidate_count",
            "heterogeneous_pair_count",
        ):
            totals[key] += int(stats.get(key, 0) or 0)
        for pair_idx, count in dict(stats.get("selected_pair_histogram") or {}).items():
            key = str(pair_idx)
            totals["selected_pair_histogram"][key] = (
                int(totals["selected_pair_histogram"].get(key, 0)) + int(count)
            )
        for pattern, count in dict(stats.get("output_pattern_counts") or {}).items():
            if pattern in totals["output_pattern_counts"]:
                totals["output_pattern_counts"][pattern] += int(count)


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

    # --- Apply semantic residual pruning (V1/V2 validation-only, V3 true drop) ---
    if args.enable_vtc:
        from models import apply_vtc_dy
        pruner_kwargs = dict(
            retention_ratio=args.vtc_retention_ratio,
            semantic_i_threshold=args.vtc_semantic_i_threshold,
            p_frame_min_tokens=args.vtc_p_frame_min_tokens,
            fill_mode=args.semantic_fill_mode,
            window_size=args.semantic_window_size,
            warmup_frames=args.semantic_warmup_frames,
        )
        pruner_kwargs.update(
            enable_vit_chunking=args.enable_vit_chunking,
            vit_chunk_frames=args.vit_chunk_frames,
            vit_chunk_staging_device=args.vit_chunk_staging_device,
            vit_chunk_collect_stats=args.vit_chunk_collect_stats,
            vit_chunk_verify_equivalence=args.vit_chunk_verify_equivalence,
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
    mosaic_pair_totals = _empty_mosaic_pair_totals()
    dyrso_i_pair_totals = _empty_dyrso_i_pair_totals()

    data = json.load(open(args.gt_file, 'r'))
    gt_questions = []
    for item in data:

        question = item['question']
        option = [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(item["candidates"])]
        qid = item['id']
        # frame_idx = item.get('frame_idx',[])
        video_id = item["video_path"].split('.')[0]
        answer_id = chr(ord("A")+item["correct_choice"])
        answer = item["candidates"][item["correct_choice"]]

        duration_group = item['duration_group']

        index2ans = {}
        for i in range(len(item["candidates"])):
            idx = chr(ord("A")+i)
            ans = item["candidates"][i]
            index2ans[idx] = ans

        gt_questions.append({
            'qid': qid, 
            'question': question, 
            'option': option, 
            'video_id': video_id, 
            'answer_id': answer_id, 
            'answer': answer, 
            'index2ans': index2ans,
            'duration_group': duration_group,
            # "frame_idx": frame_idx,
        })
        
    random.seed(42)
    random.shuffle(gt_questions)

    questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    experiment_config = vars(args).copy()
    with open(os.path.join(args.output_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=2, sort_keys=True)

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
                    if "id" in rec:
                        existing_ids.add(rec["id"])
                except json.JSONDecodeError:
                    continue
    ans_file = open(answers_file, "a", encoding="utf-8")
    
    # questions = questions[:2]

    for line in tqdm(questions):
        qid = line["qid"]
     
        if qid in existing_ids:
            continue
        answer = line["answer"]
        video_id = line["video_id"]
        answer_id = line["answer_id"]
        option = line["option"]
        index2ans = line["index2ans"]
        duration_group = line['duration_group']

        question = build_longvideobench_prompt(line["question"], option)

        sample_set = {
            "id": qid, 
            "video_id": video_id,
            "question": question, 
            "answer": answer, 
            "answer_id": answer_id, 
            'duration_group': duration_group,
        }

        video_path = os.path.join(args.video_dir, f'{video_id}.mp4')

        if args.debug:
            print(video_path)

        # Check if the video exists
        if not os.path.exists(video_path):
            if args.debug:
                print(f'Miss video {video_path}')
            continue

        # if args.uniform.lower() == "true":
        #     keyframes = []
        # else:
        #     keyframes = line["frame_idx"]
        #     sample_set["frame_idx"]= keyframes
        # print(keyframes)

        video_content = {
            "type": "video",
            "video": video_path,
            # "keyframes":keyframes,
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

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
                    print(f"[Video {i}] raw_frames={v.shape[0]} shape={list(v.shape)} fps={video_metadatas[i].get('fps','?')} total_frames={video_metadatas[i].get('total_num_frames','?')}")
        else:
            video_metadatas = None
        _accumulate_mosaic_pair_totals(mosaic_pair_totals, video_metadatas)
        _accumulate_dyrso_i_pair_totals(dyrso_i_pair_totals, video_metadatas)


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
            "top_p": float(os.getenv("top_p", 0.8)),
            "top_k": int(os.getenv("top_k", 20)),
            "temperature": float(os.getenv("temperature", 0.7)),
            "repetition_penalty": float(os.getenv("repetition_penalty", 1.0)),
            "max_new_tokens": int(os.getenv("out_seq_length", 512)),  # 16384 可能过大，这里给个更安全默认
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
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        parsed_pred = parse_multi_choice_response(outputs, ["A", "B", "C", "D","E"], index2ans)
        is_correct = parsed_pred == answer_id
        sample_set['acc'] = str(is_correct)
        sample_set["pred"] = outputs
        vtc_token_stats = latest_vtc_token_stats(
            model,
            fallback_video_id=video_id,
        )
        if vtc_token_stats is not None:
            sample_set["vtc_token_stats"] = vtc_token_stats
        vit_chunk_stats = getattr(model, "_vit_chunk_last_stats", None)
        if vit_chunk_stats is not None:
            sample_set["vit_chunk_stats"] = vit_chunk_stats
        ans_file.write(json.dumps(sample_set)+ "\n")
        ans_file.flush()

    ans_file.close()
    summary = {
        "config": experiment_config,
        "answers_file": answers_file,
        "vit_chunking_enabled": args.enable_vit_chunking,
        "last_vit_chunk_stats": getattr(model, "_vit_chunk_last_stats", None),
        "last_vit_chunk_equivalence": getattr(model, "_vit_chunk_equivalence_last", None),
        "mosaic_pair_stats": mosaic_pair_totals,
        "dyrso_i_pair_stats": dyrso_i_pair_totals,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", help="Directory containing video files.", required=True)
    parser.add_argument("--gt_file", help="Path to the ground truth file containing question and answer.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--attn_implementation", type=str, default="flash_attention_2"
    )

    parser.add_argument("--multi_gpu", action="store_true", default=False,
                        help="Use device_map='auto' to split model across visible GPUs (for 64-frame input)")

    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--uniform", type=str, default="false")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--max_pixels", type=int, default=0,
                        help="Optional per-frame video max_pixels. 0 keeps qwen-vl-utils dynamic default.")
    parser.add_argument("--video_load_branch", type=str, default="default",
                        choices=["default", "mosaic"],
                        help="Deprecated compatibility option; the branch is derived from --frame_extract_version.")
    parser.add_argument("--mosaic_source_frames", type=int, default=8,
                        help="Number of source frames sampled per group when --video_load_branch mosaic is used.")
    parser.add_argument("--mosaic_read_batch_size", type=int, default=256,
                        help="Unique frame indices per decord get_batch call when --video_load_branch mosaic is used.")
    parser.add_argument("--frame_extract_version", type=str, default="v1",
                        choices=["v0", "v1"],
                        help="Frame extraction version. v0 uses uniform sampling; v1 uses online DyRSO extraction.")
    parser.add_argument("--mosaic_packing_order", type=str, default=None,
                        choices=["qwen3_2i2p", "miniset_1i1p"],
                        help="Optional low-level per-group mosaic output order override.")
    parser.add_argument("--mosaic_pair_pack_order", type=str, default="sequential",
                        choices=["sequential", "pair_aligned"],
                        help="How each pair of P mosaics assigns source frames to matching tile positions.")
    parser.add_argument("--mosaic_require_homogeneous_pair", action=argparse.BooleanOptionalAction, default=False,
                        help="Require every Qwen3 temporal pair to be full+full or mosaic+mosaic.")
    parser.add_argument("--mosaic_strict_pairing", action=argparse.BooleanOptionalAction, default=True,
                        help="Raise on unsafe pair-aligned mosaic tails or misaligned tile pairs.")
    parser.add_argument("--mosaic_dump_pair_mapping", action="store_true", default=False,
                        help="Print one concise pair-aligned mosaic mapping block per sample.")
    parser.add_argument("--dyrso_i_pair_mode", type=str, default="fixed",
                        choices=["fixed", "hevc_bitcost", "hevc_tile_sort"],
                        help="DyRSO full-resolution I-I pair selection mode.")
    parser.add_argument("--dyrso_hevc_pair_max_weight", type=float, default=0.3,
                        help="Weight for max normalized frame bitcost in HEVC pair score.")
    parser.add_argument("--dyrso_i_pair_candidates", type=str, default=None,
                        help="Comma-separated candidate I-pair indices. Empty uses first/middle/last.")
    parser.add_argument("--dyrso_i_pair_strict", action=argparse.BooleanOptionalAction, default=True,
                        help="Raise on HEVC I-pair alignment/score failures.")
    parser.add_argument("--dyrso_dump_i_pair_selection", action="store_true", default=False,
                        help="Print concise HEVC-guided I-pair selection logs.")

    # --- Semantic Residual Pruning (V1/V2 validation-only, V3 true drop, V3.1 diagnostics) ---
    parser.add_argument("--enable_vtc", action="store_true", default=False,
                        help="Enable HEVC-inspired semantic residual token pruning")
    parser.add_argument("--semantic_version", "--semantic_pruning_version", type=str, default="v5.0.0",
                        choices=["v5.0.0", "v500"],
                        help="Pruner version.")
    parser.add_argument("--vtc_retention_ratio", type=float, default=0.5,
                        help="Fraction of tokens to keep. V1-V3/V3.4 use P-frame keep ratio; V4+ uses overall visual-token keep ratio.")
    parser.add_argument("--vtc_semantic_i_threshold", type=float, default=2.0,
                        help="Relative energy multiplier for I-frame detection (default: 2.0x)")
    parser.add_argument("--vtc_p_frame_min_tokens", type=int, default=8,
                        help="Minimum tokens to keep per frame (default: 8)")
    parser.add_argument("--semantic_max_iframes", type=int, default=-1,
                        help="V4.3: maximum full I-frames; -1 auto-computes from budget.")
    parser.add_argument("--semantic_min_iframe_gap", type=int, default=-1,
                        help="V4.3: minimum temporal-step gap between admitted I-frames; -1 auto.")
    parser.add_argument("--semantic_budget_infeasible_policy", type=str, default="adjust_p_min",
                        choices=["adjust_p_min", "error"],
                        help="V4.3: behavior when p_min_keep is infeasible under hard budget.")
    parser.add_argument("--semantic_enable_iframe_prefix_quota", action=argparse.BooleanOptionalAction, default=True,
                        help="V4.3: enable temporal prefix cap for admitted I-frames.")
    parser.add_argument("--vtc_token_allocation_mode", type=str, default="dyta",
                        choices=["uniform", "residual_mean", "dyta"],
                        help="V4.7.2: P-frame budget allocation mode.")
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
    parser.add_argument("--vtc_i_frame_compression", type=str, default="lightweight",
                        choices=["none", "lightweight"])
    parser.add_argument("--vtc_i_frame_retention_ratio", type=float, default=0.9)
    parser.add_argument("--semantic_first_iframe_keep_ratio", type=float, default=None)
    parser.add_argument("--semantic_dynamic_iframe_keep_ratio", type=float, default=None)
    parser.add_argument("--semantic_iframe_min_keep_ratio", type=float, default=0.8)
    parser.add_argument("--semantic_iframe_anchor_ratio", type=float, default=0.2)
    parser.add_argument("--semantic_iframe_selection_mode", type=str, default="spatial_anchor_salient", choices=["spatial_anchor_salient", "salient_only", "random"])
    parser.add_argument("--semantic_iframe_anchor_grid", type=str, default="4x6")
    parser.add_argument("--vtc_i_frame_residual_weight", type=float, default=0.5)
    parser.add_argument("--semantic_iframe_compress_when_ratio_one", action="store_true", default=False)
    parser.add_argument("--semantic_fill_mode", type=str, default="zero",
                        choices=["zero", "mean", "previous"],
                        help="Fill mode for V1/V2. Ignored by V3, which drops tokens.")
    parser.add_argument("--vtc_token_selection_mode", type=str, default="sc_topk",
                        choices=["sc_topk", "global_topk"],
                        help="V4.7.1 P-frame token selection mode.")
    parser.add_argument("--vtc_region_grid_h", type=int, default=4,
                        help="V4.7.1 spatial coverage grid rows.")
    parser.add_argument("--vtc_region_grid_w", type=int, default=4,
                        help="V4.7.1 spatial coverage grid columns.")
    parser.add_argument("--vtc_coverage_ratio", type=float, default=0.20,
                        help="V4.7.1 fraction of P-frame budget reserved for spatial coverage.")
    parser.add_argument("--vtc_region_min_tokens", type=int, default=1,
                        help="V4.7.1 minimum per-region target for spatial coverage allocation.")
    parser.add_argument("--semantic_window_size", type=int, default=8,
                        help="Rolling window size for keyframe detection (default: 8)")
    parser.add_argument("--semantic_warmup_frames", type=int, default=2,
                        help="Frames before keyframe detection activates (default: 2)")
    parser.add_argument("--enable_vit_chunking", action="store_true", default=False,
                        help="Encode video pixel patches through the ViT in temporal chunks before one global pruning pass.")
    parser.add_argument("--vit_chunk_frames", type=int, default=32,
                        help="Raw video frames per ViT chunk. Must be divisible by the model temporal patch size.")
    parser.add_argument("--vit_chunk_staging_device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Where to stage per-chunk ViT features before concatenation.")
    parser.add_argument("--vit_chunk_collect_stats", action="store_true", default=False,
                        help="Collect per-chunk timing and CUDA peak memory stats.")
    parser.add_argument("--vit_chunk_verify_equivalence", action="store_true", default=False,
                        help="Also run the original unchunked ViT path and compare features. Intended for small inputs.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug-level log output (default: progress bar only).")

    args = parser.parse_args()
    args.video_load_branch = "default" if args.frame_extract_version == "v0" else "mosaic"
    if not 0.0 <= float(args.dyrso_hevc_pair_max_weight) <= 1.0:
        raise ValueError("--dyrso_hevc_pair_max_weight must be in [0,1]")
    if args.mosaic_source_frames <= 0:
        raise ValueError("--mosaic_source_frames must be positive")
    if args.mosaic_read_batch_size <= 0:
        raise ValueError("--mosaic_read_batch_size must be positive")
    if args.vtc_region_grid_h <= 0 or args.vtc_region_grid_w <= 0:
        raise ValueError("--vtc_region_grid_h/w must be positive")
    if args.vtc_region_min_tokens < 0:
        raise ValueError("--vtc_region_min_tokens must be non-negative")
    return args


if __name__ == "__main__":
    args = parse_args()

    run_inference(args)
