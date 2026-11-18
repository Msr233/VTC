# Implementation details

This release is based on the submission implementation, with paper-aligned
names and execution optimizations. The formulas below retain the evaluated
discrete budgeting choices rather than silently changing the algorithm.

## Semantic I/P frames

The first temporal unit is an I frame. Subsequent units use identity
prediction from the immediately preceding unit's complete features.
Per-token importance is the Euclidean residual norm. A unit becomes I when
its mean residual exceeds twice the historical mean, after at least two
historical residuals are available. The eight-entry rolling history resets
after an I decision. Full reference features survive token removal.

Bitstream I/P/B types and the full-image/canvas layout are separate from
these semantic I/P decisions. The release omits codec-I/P classification
and alternate feature-reference ablations.

## Codec priors

`ffprobe` provides per-frame packet bytes (`pkt_size`), converted to bits.
These are coded-frame packet-size priors, not isolated residual bits.
H.264 and HEVC are supported. The default computes `log1p(bits/(H*W))`,
then median/MAD normalization separately for codec P and B types. Scores
are clipped to [-5,5] and passed through Softplus; a near-zero MAD produces
a neutral score. Codec I frames are excluded from the default prior.

Every pair of Qwen3 input images forms one temporal unit. Codec windows
are bounded by midpoints between neighboring unit centers, with the first
and last extending to the video boundaries. The interval score combines
Mean and P90 with weights alpha and 1-alpha. Default alpha is 0.5.
Missing interval P/B frames use the nearest P/B frame, or a neutral score
when none exists. Known extraction/alignment failures warn and use uniform
P budgets, unless strict mode is selected.

## Integer budgets and selection

VTC retains all I tokens and evenly distributes the remaining feasible
budget over P frames. VTCdy requests `ceil(0.9*N)` I tokens with a minimum
retention of 0.8, then transfers released tokens to P frames within their
capacity. An all-I chunk retains complete references. Mandatory I frames
can make the requested overall retention infeasible; actual counts are
reported rather than fabricated to match the requested ratio.

The interval complexity is rounded to six decimal places before allocation,
as in the base implementation. The P-frame allocator uses its square root, a
minimum target of eight tokens and a cap of twice the average P budget,
subject to frame capacity and feasibility. Saturated allocations are
redistributed; largest-remainder rounding conserves the feasible integer
budget. These bounded weights and the I-frame `ceil` differ from the
paper's compact linear/round formulas.

I-frame importance is deviation from its mean feature, mixed equally
with temporal residual for later I frames. The first I frame uses only
within-frame deviation. Its 4x6 floor-partitioned grid supplies at most
one anchor per nonempty region before global completion. P-frame FRS
uses a 4x4 `tensor_split` grid and reserves
`min(K,max(nonempty_regions,floor(0.2*K)))` selections for region coverage.
Thus I and P selectors share the coverage principle but have distinct
region quotas and integer boundaries. This detail is retained from the
experimental code.

## Streaming and GoP inputs

Chunk allocation uses the remaining video target and future temporal
slots, while residual history persists across chunks. It is not global
allocation over all video features; keep chunk size fixed in comparisons.

The fixed sequential input plan packs two full images followed by two
2x2 canvases, each containing four half-size images. Qwen3 temporal fusion
maps this to two units, corresponding to one full and four half-size
images per unit at the input level. The default 64 images count model
inputs, not unique decoded source frames.

VideoMME retains its original direct uniform video-resize path. Other
uniform runners and the GoP path retain the original image-style resize
followed by the video token-budget resize. OpenCV area interpolation is
used for canvas tiles when available, otherwise PIL bilinear is used.
No private extracted-image caches or adaptive packing variants are required.

## Evaluation

Dataset-specific prompts, answer parsing and seed-42 strided question
sharding are retained. Some parsers use seeded random fallback when no
choice is recognized. Generation is greedy. Output limits follow the
original runners: 512 tokens for VideoMME/LongVideoBench and 16 for
MLVU/LVBench. Changing sharding, resolution, attention backend or chunk
size can affect predictions.

The shared configuration covers method defaults. Dataset-specific tuning
for every paper table is not inferred or claimed by the launcher.
