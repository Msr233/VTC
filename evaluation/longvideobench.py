"""Qwen3-VL evaluation: longvideobench."""

from evaluation.common import parse_common_args, configure_model, prepare_video_context
from transformers import AutoModelForImageTextToText, AutoProcessor
from evaluation.attention_backend import resolve_attention_backend
from vtc.gop_packing import process_gop_vision_info
from vtc.video_io import process_vision_info
from evaluation.sampling_context import clear_vtc_video_contexts
import os
import torch
import numpy as np


def get_chunk(lst, n, k):
    return lst[k::n]


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
    response = " " + response + " "
    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
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
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]
    return pred_index


import json
import random
from tqdm import tqdm
from evaluation.token_stats import latest_vtc_token_stats

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
    data = json.load(open(args.gt_file, "r"))
    gt_questions = []
    for item in data:
        question = item["question"]
        option = [
            ". ".join([chr(ord("A") + i), candidate])
            for i, candidate in enumerate(item["candidates"])
        ]
        qid = item["id"]
        video_id = item["video_path"].split(".")[0]
        answer_id = chr(ord("A") + item["correct_choice"])
        answer = item["candidates"][item["correct_choice"]]
        duration_group = item["duration_group"]
        index2ans = {}
        for i in range(len(item["candidates"])):
            idx = chr(ord("A") + i)
            ans = item["candidates"][i]
            index2ans[idx] = ans
        gt_questions.append(
            {
                "qid": qid,
                "question": question,
                "option": option,
                "video_id": video_id,
                "answer_id": answer_id,
                "answer": answer,
                "index2ans": index2ans,
                "duration_group": duration_group,
            }
        )
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
    with open(
        os.path.join(args.output_dir, "experiment_config.json"), "w", encoding="utf-8"
    ) as f:
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
    for line in tqdm(questions):
        qid = line["qid"]
        if qid in existing_ids:
            continue
        answer = line["answer"]
        video_id = line["video_id"]
        answer_id = line["answer_id"]
        option = line["option"]
        index2ans = line["index2ans"]
        duration_group = line["duration_group"]
        question = build_longvideobench_prompt(line["question"], option)
        sample_set = {
            "id": qid,
            "video_id": video_id,
            "question": question,
            "answer": answer,
            "answer_id": answer_id,
            "duration_group": duration_group,
        }
        video_path = os.path.join(args.video_dir, f"{video_id}.mp4")
        if args.debug:
            print(video_path)
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
            "top_p": float(os.getenv("top_p", 0.8)),
            "top_k": int(os.getenv("top_k", 20)),
            "temperature": float(os.getenv("temperature", 0.7)),
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
            prepare_video_context(model, args, [video_path], video_metadatas, inputs)
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
            outputs, ["A", "B", "C", "D", "E"], index2ans
        )
        is_correct = parsed_pred == answer_id
        sample_set["acc"] = str(is_correct)
        sample_set["pred"] = outputs
        sample_set["codec_preload_ms"] = getattr(model, "_vtc_codec_preload_ms", 0.0)
        if args.log_frame_tokens and args.method != "none":
            sample_set["frame_tokens"] = model._vtc_last_stats[0]["frame_info"]
        vtc_token_stats = latest_vtc_token_stats(model, fallback_video_id=video_id)
        if vtc_token_stats is not None:
            sample_set["vtc_token_stats"] = vtc_token_stats
        vit_chunk_stats = getattr(model, "_vit_chunk_last_stats", None)
        if vit_chunk_stats is not None:
            sample_set["vit_chunk_stats"] = vit_chunk_stats
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()
    ans_file.close()
    summary = {
        "config": experiment_config,
        "answers_file": answers_file,
        "vit_chunking_enabled": (args.vit_chunk_frames > 0),
        "last_vit_chunk_stats": getattr(model, "_vit_chunk_last_stats", None),
    }
    with open(
        os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    args = parse_common_args("longvideobench")
    run_inference(args)
