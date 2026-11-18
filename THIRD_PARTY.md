# Third-party components

- Qwen3-VL is supplied through the separately installed `transformers`
  package and separately obtained model weights. The integration imports
  `Qwen3VLModelOutputWithPast` from Transformers 4.57.0 and patches the
  loaded model at runtime.
- Video/image IO and resize helpers are adapted from the Qwen-VL utility
  interface used by the experimental code. The release retains the utility
  limits, sampling behavior, and preprocessing needed by the runners.
- The LongVideoBench multiple-choice response parser retains its source
  reference to the public MMMU evaluation utility:
  <https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10>.
- PyTorch, Torchvision, Transformers, Decord, NumPy, Pillow, OpenCV,
  Requests, and FFmpeg/ffprobe are external dependencies. Optional
  FlashAttention is installed separately. Their own licenses apply.

No model weights, benchmark data, or external repositories are bundled.
