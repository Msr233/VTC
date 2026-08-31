import unittest
import inspect

import torch

from eval.qwen_vl_utils_new_2 import (
    build_mosaic_frame_plan,
    validate_homogeneous_temporal_pairs,
)
from models.visual_token_coding import VisualTokenCoding


class VTCDyTest(unittest.TestCase):
    def test_submission_pruner_has_no_legacy_diagnostic_interface(self):
        removed = {
            "predict_mode",
            "local_window_size",
            "save_energy_heatmap",
            "heatmap_dir",
            "save_heatmap_png",
            "save_residual_csv",
            "residual_csv_path",
            "codec_dump_alignment",
            "collect_residual_diagnostics",
            "collect_detailed_energy_stats",
            "collect_spatial_diagnostics",
        }
        parameters = inspect.signature(VisualTokenCoding.__init__).parameters
        self.assertTrue(removed.isdisjoint(parameters))

        pruner = VisualTokenCoding(
            retention_ratio=0.75,
            p_frame_min_tokens=1,
            token_allocation_mode="uniform",
            codec_bitcost_enabled=False,
            spatial_merge_size=1,
            iframe_compression_mode="none",
            log_stats=False,
        )
        _, stats = pruner.forward(
            torch.arange(32, dtype=torch.float32).reshape(8, 4),
            torch.tensor([2, 2, 2]),
        )
        for key in (
            "predict_mode",
            "residual_stats",
            "codec_semantic_pearson",
            "codec_semantic_spearman",
            "mean_selected_region_entropy",
        ):
            self.assertNotIn(key, stats)
        for frame in stats["frame_info"]:
            self.assertNotIn("semantic_residual_p90", frame)
            self.assertNotIn("feature_contrast_p90", frame)

    def test_dyrso_matches_v472_fixed_iipp_input(self):
        _, plans, _ = build_mosaic_frame_plan(
            total_num_frames=100,
            nframes=8,
            dyrso_i_pair_mode="fixed",
            mosaic_pair_pack_order="sequential",
            dyrso_i_pair_strict=True,
        )
        self.assertGreater(len(plans), 0)
        for plan in plans:
            selection = plan["dyrso_i_pair_selection"]
            self.assertEqual(selection["mode"], "fixed")
            self.assertEqual(selection["selected_pair_index"], 0)
            self.assertEqual(plan["original_frames"], plan["logical_indices"][:2])
            self.assertTrue(plan["pair_alignment_valid"])
            for tile_pair in plan["pair_tile_pairs"]:
                self.assertGreater(
                    tile_pair["second_frame_index"],
                    tile_pair["first_frame_index"],
                )
        validate_homogeneous_temporal_pairs(
            ["full", "full", "mosaic", "mosaic"]
        )

    def test_v5_codec_guided_full_pair_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            build_mosaic_frame_plan(
                total_num_frames=100,
                nframes=4,
                dyrso_i_pair_mode="dyrso",
            )

    def test_mixed_resolution_temporal_pairs_are_rejected(self):
        for pattern in ("IPPI", "PIIP", "IPIP", "PIPI"):
            input_types = ["full" if value == "I" else "mosaic" for value in pattern]
            with self.assertRaises(AssertionError):
                validate_homogeneous_temporal_pairs(input_types)

    def test_short_video_duplicate_samples_remain_pair_safe(self):
        _, plans, _ = build_mosaic_frame_plan(
            total_num_frames=2,
            nframes=4,
            dyrso_i_pair_mode="fixed",
            dyrso_i_pair_strict=True,
        )
        selection = plans[0]["dyrso_i_pair_selection"]
        self.assertEqual(selection["mode"], "fixed")
        self.assertEqual(selection["selected_pair_index"], 0)
        self.assertTrue(plans[0]["pair_alignment_valid"])


if __name__ == "__main__":
    unittest.main()
