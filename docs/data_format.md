# Benchmark input format

Each `--gt-file` is a UTF-8 JSON **array**, not JSON Lines. Examples below
illustrate field names only and are not benchmark samples. Keep the original
benchmark questions, choices, answers, and evaluation split.

## VideoMME

```json
[
  {
    "question_id": "example_question",
    "videoID": "example_video",
    "duration": "short",
    "question": "Example question?",
    "options": ["A. Option one", "B. Option two", "C. Option three", "D. Option four"],
    "answer": "A"
  }
]
```

Videos are located at `<video-dir>/<videoID>.mp4`. Use the multiple-choice
evaluation annotations. The runner does not load subtitle files; it retains
the original prompt wording but supplies video and question text only.

## MLVU

```json
[
  {
    "qid": "example_question",
    "category": "example_category",
    "video": "example_video.mp4",
    "question": "Example question?",
    "candidates": ["Option one", "Option two", "Option three", "Option four"],
    "answer": "Option one",
    "duration": 100,
    "question_type": "example_task"
  }
]
```

`answer` is the exact correct candidate **text**. Videos are located at
`<video-dir>/<category>/<video>`. Merge the multiple-choice evaluation
annotations into one array and assign a unique `qid` to every question.
Use the full evaluation set (2174 questions for the paper's MLVU protocol);
the release applies no private exclusion list. Report total accuracy.

## LVBench

```json
[
  {
    "key": "example_video",
    "uid": "example_question",
    "video_path": "example_video.mp4",
    "question": "Example question?\n(A) Option one\n(B) Option two\n(C) Option three\n(D) Option four",
    "answer": "A",
    "question_type": [],
    "type": "example_video_type",
    "time_reference": ""
  }
]
```

Videos are located at `<video-dir>/<video_path>`. If `video_path` is omitted,
the runner uses `<key>.mp4`. The runner forms question IDs as `<key>_<uid>`.
Choices remain embedded in `question`.

## LongVideoBench

```json
[
  {
    "id": "example_question",
    "video_path": "example_video.mp4",
    "question": "Example question?",
    "candidates": ["Option one", "Option two", "Option three", "Option four"],
    "correct_choice": 0,
    "duration_group": 15
  }
]
```

`correct_choice` is zero-based. Videos are located at
`<video-dir>/<video_path>`; the inherited runner expects flat `.mp4` names.
Use the validation video-only protocol; subtitle text is not supplied.
The prompt ends with:

```text
Please answer directly with only the letter of the correct option and nothing else.
```

## Outputs and incomplete evaluation

Predictions contain `id`, `video_id`, `answer_id`, `pred`, and `acc`, plus
dataset metadata and visual token statistics. Missing videos are logged and
skipped by the original runners. The summary command checks predictions
against all IDs in the supplied annotation array and reports missing IDs.
Do not mix partial runs, different configurations, or duplicate question
shards when aggregating results.
