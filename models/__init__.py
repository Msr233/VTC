from __future__ import annotations

from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

import torch

from .chunked_video_encoder import (
    ChunkedVideoEncoderConfig,
    encode_and_prune_video_in_chunks,
)
from .visual_token_coding import VisualTokenCoding
from .vtc_dy import VTCDy


def _configure_vtc_version(version: str, pruner_kwargs: Dict[str, Any]) -> str:
    normalized = {
        "v500": "v5.0.0",
        "v5": "v5.0.0",
    }.get(
        str(version),
        str(version),
    )
    if normalized != "v5.0.0":
        raise ValueError(
            "The submission package only supports v5.0.0; "
            f"got version={version!r}"
        )
    pruner_kwargs["semantic_version"] = normalized
    pruner_kwargs.setdefault("semantic_frame_type_source", "residual")
    return normalized


def apply_vtc_dy(model, version: str = "v5.0.0", **pruner_kwargs):
    """Apply the VTC_Dy path to Qwen3-VL."""
    normalized_version = _configure_vtc_version(version, pruner_kwargs)
    inner = model.model
    vit_chunk_config = ChunkedVideoEncoderConfig(
        enabled=bool(pruner_kwargs.pop("enable_vit_chunking", False)),
        chunk_frames=int(pruner_kwargs.pop("vit_chunk_frames", 32)),
        staging_device=str(pruner_kwargs.pop("vit_chunk_staging_device", "cuda")),
        collect_stats=bool(pruner_kwargs.pop("vit_chunk_collect_stats", False)),
        verify_shapes=True,
    )
    pruner_kwargs.pop("vit_chunk_verify_equivalence", None)
    pruner = VTCDy(**pruner_kwargs)
    inner._vtc = pruner
    _patch_visual_token_coding(model, inner, pruner, vit_chunk_config)
    print(
        f"[VTC_Dy] Applied Visual Token Coding {normalized_version} to Qwen3VLModel.\n"
        f"         requested_version={version} retention_ratio={pruner.retention_ratio} "
        f"token_allocation={pruner.token_allocation_mode} "
        f"i_frame_compression={pruner.iframe_compression_mode} "
        f"token_selection={pruner.token_selection_mode} "
        f"vit_chunking={vit_chunk_config.enabled}"
    )
    return pruner


def _attach_semantic_log_context(
    all_stats: List[Dict[str, Any]],
    pruner: VisualTokenCoding,
) -> None:
    context = getattr(pruner, "_semantic_log_context", None)
    if not isinstance(context, dict):
        return
    for stats in all_stats:
        for key in ("sample_id", "question_id", "video_id"):
            value = context.get(key)
            if value is not None:
                stats[key] = value


def _log_pruning_stats(
    all_stats: List[Dict[str, Any]],
    pruner: VisualTokenCoding,
) -> None:
    if not all_stats or not pruner.log_stats:
        return
    _attach_semantic_log_context(all_stats, pruner)
    for stats in all_stats:
        print(pruner.format_stats(stats))


def _patch_visual_token_coding(
    model,
    inner,
    pruner: VisualTokenCoding,
    vit_chunk_config: ChunkedVideoEncoderConfig,
) -> None:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModelOutputWithPast,
    )

    original_get_image_features = inner.get_image_features
    original_forward = inner.forward
    original_prepare = getattr(model, "prepare_inputs_for_generation", None)

    def compute_pruned_visual(self, pixel_values, grid_thw, is_video: bool = False):
        chunk_stats = None
        if is_video and vit_chunk_config.enabled:
            temporal_patch_size = int(
                getattr(self.visual.patch_embed, "temporal_patch_size", 1)
            )
            spatial_merge_size = int(getattr(self.visual, "spatial_merge_size", 1))
            (
                image_embeds,
                deepstack_image_embeds,
                keep_masks,
                all_stats,
            ) = encode_and_prune_video_in_chunks(
                self.visual,
                pixel_values,
                grid_thw,
                pruner,
                temporal_patch_size=temporal_patch_size,
                spatial_merge_size=spatial_merge_size,
                chunk_frames=vit_chunk_config.chunk_frames,
                staging_device=vit_chunk_config.staging_device,
                collect_stats=vit_chunk_config.collect_stats,
                verify_shapes=vit_chunk_config.verify_shapes,
            )
            self._vit_chunk_last_stats = [
                stats.get("vit_chunk_stats") for stats in all_stats
            ]
            model._vit_chunk_last_stats = self._vit_chunk_last_stats
            _log_pruning_stats(all_stats, pruner)
            return image_embeds, deepstack_image_embeds, keep_masks, all_stats

        image_embeds, deepstack_image_embeds = original_get_image_features(
            pixel_values,
            grid_thw,
        )
        split_sizes = [embeds.shape[0] for embeds in image_embeds]
        deepstack_chunks = None
        if deepstack_image_embeds is not None:
            deepstack_chunks = [
                torch.split(layer, split_sizes)
                for layer in deepstack_image_embeds
            ]

        pruned_embeds: List[torch.Tensor] = []
        keep_masks: List[torch.Tensor] = []
        all_stats: List[Dict[str, Any]] = []
        pruned_deepstack_parts: Optional[List[List[torch.Tensor]]] = (
            [[] for _ in deepstack_image_embeds]
            if deepstack_image_embeds is not None
            else None
        )

        for idx, embeds in enumerate(image_embeds):
            pruned, stats = pruner.forward(embeds, grid_thw[idx])
            keep_mask = stats["keep_mask"]
            if keep_mask.numel() != embeds.shape[0]:
                raise ValueError(
                    "[VTC_Dy] keep_mask length does not match full visual "
                    f"tokens: mask={keep_mask.numel()}, tokens={embeds.shape[0]}"
                )
            if chunk_stats is not None:
                stats["vit_chunk_stats"] = chunk_stats[idx]
            pruned_embeds.append(pruned)
            keep_masks.append(keep_mask)
            all_stats.append(stats)

            if pruned_deepstack_parts is not None and deepstack_chunks is not None:
                for layer_idx, layer_chunks in enumerate(deepstack_chunks):
                    pruned_deepstack_parts[layer_idx].append(
                        layer_chunks[idx][keep_mask]
                    )

        if pruned_deepstack_parts is None:
            pruned_deepstack = deepstack_image_embeds
        else:
            pruned_deepstack = [
                torch.cat(parts, dim=0)
                if parts
                else deepstack_image_embeds[layer_idx].new_empty(
                    (0, deepstack_image_embeds[layer_idx].shape[-1])
                )
                for layer_idx, parts in enumerate(pruned_deepstack_parts)
            ]

        _log_pruning_stats(all_stats, pruner)
        return tuple(pruned_embeds), pruned_deepstack, keep_masks, all_stats

    def patched_get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        pruned_embeds, pruned_deepstack, keep_masks, all_stats = compute_pruned_visual(
            self,
            pixel_values,
            image_grid_thw,
        )
        self._vtc_last_keep_masks = keep_masks
        self._vtc_last_stats = all_stats
        model._vtc_last_stats = all_stats
        return pruned_embeds, pruned_deepstack

    def apply_visual_keep(
        input_ids: torch.LongTensor,
        seq_keep: torch.Tensor,
        token_id: int,
        keep_masks: Optional[List[torch.Tensor]],
        label: str,
    ) -> None:
        if not keep_masks:
            return
        flat_keep = torch.cat([m.to(input_ids.device) for m in keep_masks]).bool()
        positions = torch.nonzero(input_ids == token_id, as_tuple=False)
        if positions.shape[0] != flat_keep.numel():
            raise ValueError(
                f"[VTC_Dy] {label} placeholder count does not match "
                f"keep mask length: tokens={positions.shape[0]} "
                f"mask={flat_keep.numel()}"
            )
        seq_keep[positions[:, 0], positions[:, 1]] = flat_keep

    def build_pruned_sequence(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.LongTensor],
        video_grid_thw: Optional[torch.LongTensor],
        image_keep_masks: Optional[List[torch.Tensor]],
        video_keep_masks: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attention_mask_tensor = attention_mask
        if isinstance(attention_mask_tensor, dict):
            attention_mask_tensor = attention_mask_tensor["full_attention"]
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(
                attention_mask_tensor[:, 0],
                dim1=1,
                dim2=2,
            )
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(
                    attention_mask_tensor.dtype
                ).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        original_position_ids, _ = self.get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask=attention_mask_tensor,
        )

        seq_keep = torch.ones_like(input_ids, dtype=torch.bool)
        apply_visual_keep(
            input_ids,
            seq_keep,
            self.config.image_token_id,
            image_keep_masks,
            "image",
        )
        apply_visual_keep(
            input_ids,
            seq_keep,
            self.config.video_token_id,
            video_keep_masks,
            "video",
        )

        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 2:
            seq_keep &= attention_mask_tensor.to(input_ids.device).bool()

        batch_size = input_ids.shape[0]
        selected = [
            torch.nonzero(seq_keep[b], as_tuple=False).squeeze(1)
            for b in range(batch_size)
        ]
        lengths = [idx.numel() for idx in selected]
        max_len = max(lengths)

        pad_token_id = getattr(self.config, "pad_token_id", None)
        if pad_token_id is None and hasattr(self.config, "text_config"):
            pad_token_id = getattr(self.config.text_config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

        new_input_ids = input_ids.new_full((batch_size, max_len), pad_token_id)
        new_attention_mask = input_ids.new_zeros((batch_size, max_len))
        new_position_ids = original_position_ids.new_zeros((3, batch_size, max_len))

        rope_deltas = []
        for b, idx in enumerate(selected):
            length = idx.numel()
            new_input_ids[b, :length] = input_ids[b, idx]
            new_attention_mask[b, :length] = 1
            new_position_ids[:, b, :length] = original_position_ids[:, b, idx]
            rope_deltas.append(new_position_ids[:, b, :length].max() + 1 - length)

        self.rope_deltas = torch.stack(rope_deltas).to(input_ids.device).unsqueeze(1)
        return new_input_ids, new_attention_mask, new_position_ids, seq_keep

    def patch_generation_state(
        outer_model,
        keep_mask: torch.Tensor,
        pruned_len: int,
        original_len: int,
    ) -> None:
        outer_model._vtc_generation_state = {
            "keep_seq_mask": keep_mask.detach().cpu(),
            "original_prompt_len": int(original_len),
            "pruned_prompt_len": int(pruned_len),
            "drop": int(original_len - pruned_len),
        }

    def patched_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if (
            input_ids is None
            or inputs_embeds is not None
            or position_ids is not None
            or (pixel_values is None and pixel_values_videos is None)
        ):
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                **kwargs,
            )

        image_embeds = None
        video_embeds = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None
        image_keep_masks = None
        video_keep_masks = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds, image_keep_masks, image_stats = (
                compute_pruned_visual(self, pixel_values, image_grid_thw)
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(
                input_ids.device,
                self.get_input_embeddings().weight.dtype,
            )
        else:
            image_stats = []

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds, video_keep_masks, video_stats = (
                compute_pruned_visual(
                    self,
                    pixel_values_videos,
                    video_grid_thw,
                    is_video=True,
                )
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(
                input_ids.device,
                self.get_input_embeddings().weight.dtype,
            )
        else:
            video_stats = []

        visual_stats = list(image_stats) + list(video_stats)
        self._vtc_last_stats = visual_stats
        model._vtc_last_stats = visual_stats

        (
            new_input_ids,
            new_attention_mask,
            new_position_ids,
            seq_keep,
        ) = build_pruned_sequence(
            self,
            input_ids,
            attention_mask,
            image_grid_thw,
            video_grid_thw,
            image_keep_masks,
            video_keep_masks,
        )

        new_cache_position = None
        if cache_position is not None:
            new_cache_position = torch.arange(
                new_input_ids.shape[1],
                device=input_ids.device,
                dtype=cache_position.dtype,
            )

        patch_generation_state(
            model,
            seq_keep,
            new_input_ids.shape[1],
            input_ids.shape[1],
        )

        inputs_embeds = self.get_input_embeddings()(new_input_ids)

        image_mask = None
        video_mask = None
        if image_embeds is not None:
            image_mask = new_input_ids == self.config.image_token_id
            if image_mask.sum().item() != image_embeds.shape[0]:
                raise ValueError(
                    "[VTC_Dy] image tokens/features mismatch after drop: "
                    f"tokens={image_mask.sum().item()} features={image_embeds.shape[0]}"
                )
            image_mask_exp = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask_exp,
                image_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
            )

        if video_embeds is not None:
            video_mask = new_input_ids == self.config.video_token_id
            if video_mask.sum().item() != video_embeds.shape[0]:
                raise ValueError(
                    "[VTC_Dy] video tokens/features mismatch after drop: "
                    f"tokens={video_mask.sum().item()} features={video_embeds.shape[0]}"
                )
            video_mask_exp = video_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask_exp,
                video_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
            )

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(
                deepstack_image_embeds,
                deepstack_video_embeds,
            ):
                embed_joint = img_embed.new_zeros(
                    visual_pos_masks.sum(),
                    img_embed.shape[-1],
                ).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        outputs = self.language_model(
            input_ids=None,
            position_ids=new_position_ids,
            attention_mask=new_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=new_cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    def patched_prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        model_inputs = original_prepare(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )

        state = getattr(self, "_vtc_generation_state", None)
        if state is None or past_key_values is None:
            return model_inputs

        is_prefill = False
        if cache_position is not None and cache_position.numel() > 0:
            is_prefill = cache_position[0].item() == 0
        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        if callable(get_seq_length):
            try:
                is_prefill = is_prefill or get_seq_length() == 0
            except Exception:
                pass
        if is_prefill:
            return model_inputs

        attn = model_inputs.get("attention_mask")
        if attn is not None and attn.ndim == 2:
            orig_len = state["original_prompt_len"]
            keep = state["keep_seq_mask"].to(attn.device)
            if attn.shape[1] >= orig_len and keep.shape[0] == attn.shape[0]:
                prompt_parts = [
                    attn[b, :orig_len][keep[b]]
                    for b in range(attn.shape[0])
                ]
                prompt_len = max(part.numel() for part in prompt_parts)
                new_prompt = attn.new_zeros((attn.shape[0], prompt_len))
                for b, part in enumerate(prompt_parts):
                    new_prompt[b, : part.numel()] = part
                tail = attn[:, orig_len:]
                model_inputs["attention_mask"] = torch.cat([new_prompt, tail], dim=1)

        if model_inputs.get("cache_position") is not None:
            model_inputs["cache_position"] = torch.clamp(
                model_inputs["cache_position"] - state["drop"],
                min=0,
            )
        return model_inputs

    inner.get_image_features = MethodType(patched_get_image_features, inner)
    inner.forward = MethodType(patched_forward, inner)
    if original_prepare is not None:
        model.prepare_inputs_for_generation = MethodType(
            patched_prepare_inputs_for_generation,
            model,
        )


__all__ = [
    "VisualTokenCoding",
    "VTCDy",
    "apply_vtc_dy",
]
