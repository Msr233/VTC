"""Qwen3-VL evaluation: mlvu."""

from evaluation.common import parse_common_args, configure_model, prepare_video_context
import json
import os
import random
import re
import numpy as np
import torch
from vtc.gop_packing import process_gop_vision_info
from vtc.video_io import process_vision_info
from evaluation.token_stats import latest_vtc_token_stats
from evaluation.sampling_context import clear_vtc_video_contexts
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from evaluation.attention_backend import resolve_attention_backend


def get_chunk(lst, n, k):
    return lst[k::n]


MLVU_POST_PROMPT = "\nOnly give the best option.\nBest option: ("
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


def build_mlvu_prompt(question, candidates):
    options = []
    for idx, candidate in enumerate(candidates):
        choice = chr(ord("A") + idx)
        options.append(f"({choice}) {candidate}")
    return question.rstrip() + "\n" + "\n".join(options) + MLVU_POST_PROMPT


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
        for match in re.finditer("(?<![A-Z])([A-Z])(?=[\\)\\.\\s,:;]|$)", response):
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


def build_questions(gt_file):
    data = json.load(open(gt_file, "r", encoding="utf-8"))
    gt_questions = []
    for row in data:
        candidates = row["candidates"]
        index2ans = {
            chr(ord("A") + i): candidate for i, candidate in enumerate(candidates)
        }
        answer = row["answer"]
        answer_id = chr(ord("A") + candidates.index(answer))
        video_id = f"{row['category']}/{row['video']}"
        gt_questions.append(
            {
                "qid": row["qid"],
                "question": row["question"],
                "candidates": candidates,
                "video_id": video_id,
                "answer_id": answer_id,
                "answer": answer,
                "index2ans": index2ans,
                "duration": row.get("duration"),
                "question_type": row.get("question_type", ""),
                "category": row.get("category", ""),
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
    configure_model(model, args)
    processor = AutoProcessor.from_pretrained(args.model_path)
    gt_questions = build_questions(args.gt_file)
    random.seed(42)
    random.shuffle(gt_questions)
    questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    os.makedirs(args.output_dir, exist_ok=True)
    output_name = (
        f"{args.num_chunks}_{args.chunk_idx}"
        if args.num_chunks > 1
        else args.output_name
    )
    experiment_config = vars(args).copy()
    with open(
        os.path.join(args.output_dir, "experiment_config.json"), "w"
    ) as config_file:
        json.dump(experiment_config, config_file, indent=2, sort_keys=True)
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
            question = build_mlvu_prompt(line["question"], line["candidates"])
            sample_set = {
                "id": qid,
                "video_id": video_id,
                "question": question,
                "answer_id": line["answer_id"],
                "answer": line["answer"],
                "duration": line["duration"],
                "question_type": line["question_type"],
                "category": line["category"],
            }
            video_path = os.path.join(args.video_dir, line["video_id"])
            if not os.path.exists(video_path):
                if args.debug:
                    print(f"Miss video {video_path}")
                continue
            video_content = {
                "type": "video",
                "video": video_path,
                "nframes": args.frames,
                "method": args.method,
                "question_id": qid,
                "sample_id": qid,
            }
            if args.max_pixels and args.max_pixels > 0:
                video_content["max_pixels"] = args.max_pixels
            messages = [
                {
                    "role": "user",
                    "content": [video_content, {"type": "text", "text": question}],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if args.input_mode == "gop":
                images, videos, video_kwargs = process_gop_vision_info(
                    messages,
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                    read_batch_size=args.read_batch_size,
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
                videos, video_metadatas = (list(videos), list(video_metadatas))
                if args.debug:
                    for i, v in enumerate(videos):
                        print(
                            f"[Video {i}] raw_frames={v.shape[0]} shape={list(v.shape)} fps={video_metadatas[i].get('fps', '?')} total_frames={video_metadatas[i].get('total_num_frames', '?')}"
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
                "max_new_tokens": args.max_new_tokens,
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
                prepare_video_context(
                    model, args, [video_path], video_metadatas, inputs
                )
                generated_ids = model.generate(**inputs, **generation_kwargs)
            finally:
                clear_vtc_video_contexts(model)
                for target in (model, inner_model, vtc):
                    if target is not None and hasattr(target, "_semantic_log_context"):
                        delattr(target, "_semantic_log_context")
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            outputs = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            parsed_pred = parse_multi_choice_response(
                outputs, ["A", "B", "C", "D"], line["index2ans"]
            )
            sample_set["acc"] = str(parsed_pred == line["answer_id"])
            sample_set["parsed_pred"] = parsed_pred
            sample_set["pred"] = outputs
            sample_set["codec_preload_ms"] = getattr(
                model, "_vtc_codec_preload_ms", 0.0
            )
            if args.log_frame_tokens and args.method != "none":
                sample_set["frame_tokens"] = model._vtc_last_stats[0]["frame_info"]
            vtc_token_stats = latest_vtc_token_stats(model, fallback_video_id=video_id)
            if vtc_token_stats is not None:
                sample_set["vtc_token_stats"] = vtc_token_stats
            ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
            ans_file.flush()


if __name__ == "__main__":
    run_inference(parse_common_args("mlvu"))
