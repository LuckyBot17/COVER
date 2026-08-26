#!/usr/bin/env python3
"""Score all 15 relation candidates with COVER identity negative controls.

For each directed entity pair and candidate relation, LLaVA-1.5-7B scores the
original image and subject-, object-, and both-object blur controls in one
four-image batch.  The CSV is flushed after every candidate and is resumable
by the object-ID-safe key (image, subject ID, object ID, candidate relation).
"""

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from prepare_gqa_closedset import BASE, RELATIONS, iter_candidate_rows


DEFAULT_MODEL_PATH = "/root/autodl-tmp/llava-1.5-7b/master"
QUESTION = "Is it true that the {s} is {r} the {o} in the image?"
PROMPT_TMPL = "USER: <image>\n{question}\nASSISTANT:"
MASK_COLOR = (128, 128, 128)
STAGE_FILES = {
    "smoke": "smoke_pairs.csv",
    "pilot": "pilot_pairs.csv",
    "full": "all_pairs.csv",
}
SCORE_FIELDS = (
    "pair_id", "split", "image_id", "image_path",
    "subject_id", "subject_name", "subject_box",
    "object_id", "object_name", "object_box",
    "candidate_r", "candidate_true", "present_rels",
    "a_orig", "d_s", "d_o", "d_so", "elapsed_sec", "mask",
)

_device = None
_model = None
_processor = None
_yes_id = None
_no_id = None


def candidate_key(row):
    return (
        str(row["image_id"]), str(row["subject_id"]),
        str(row["object_id"]), str(row["candidate_r"]),
    )


def load_done_keys(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(encoding="utf-8") as f:
        return {candidate_key(row) for row in csv.DictReader(f)}


def validate_checkpoint_mask(path, requested_mask):
    """Prevent a shared checkpoint from silently mixing interventions."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open(encoding="utf-8") as f:
        modes = {row.get("mask", "") for row in csv.DictReader(f)} - {""}
    if modes and modes != {requested_mask}:
        raise ValueError(
            f"checkpoint mask mode(s) {sorted(modes)} do not match requested {requested_mask!r}"
        )


def pending_candidates(candidates, done):
    return [row for row in candidates if candidate_key(row) not in done]


def parse_box(value):
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    return tuple(int(v) for v in str(value).split("|"))


def load_pairs(path):
    with Path(path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["candidate_true"] = int(row.get("candidate_true", 0) or 0)
    return rows


def mask_image(image, boxes, mode="blur"):
    image = image.copy()
    if mode == "gray":
        draw = ImageDraw.Draw(image)
        for x, y, w, h in boxes:
            draw.rectangle([x, y, x + w, y + h], fill=MASK_COLOR)
    elif mode == "mean":
        draw = ImageDraw.Draw(image)
        for x, y, w, h in boxes:
            crop = image.crop((x, y, x + w, y + h))
            arr = np.asarray(crop, dtype=np.float32).reshape(-1, 3).mean(0).round().astype(int)
            draw.rectangle([x, y, x + w, y + h], fill=tuple(arr))
    else:
        width, height = image.size
        for x, y, w, h in boxes:
            x0, y0 = max(0, x - 20), max(0, y - 20)
            x1, y1 = min(width, x + w + 20), min(height, y + h + 20)
            patch = image.crop((x0, y0, x1, y1)).filter(
                ImageFilter.GaussianBlur(radius=max(8, int(w * 0.15)))
            )
            image.paste(patch, (x0, y0))
    return image


def load_model(model_path):
    global _device, _model, _processor, _yes_id, _no_id
    if _model is not None:
        return _model, _processor
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if _device == "cuda" else torch.float32
    _model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True
    ).eval().to(_device)
    _processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = _processor.tokenizer
    yes_ids = tokenizer("Yes", add_special_tokens=False)["input_ids"]
    no_ids = tokenizer("No", add_special_tokens=False)["input_ids"]
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise RuntimeError(f"Yes/No must be single tokens, got Yes={yes_ids}, No={no_ids}")
    _yes_id, _no_id = yes_ids[0], no_ids[0]
    print(
        f"[info] model loaded on {_device}; yes_id={_yes_id} no_id={_no_id}",
        flush=True,
    )
    return _model, _processor


def score_candidate(model, processor, image, subject, relation, obj, boxes, mask_mode="blur"):
    """Return raw log-odds and three matched negative-control differences."""
    import torch

    views = {
        "orig": image,
        "s": mask_image(image, [boxes["s"]], mode=mask_mode),
        "o": mask_image(image, [boxes["o"]], mode=mask_mode),
        "so": mask_image(image, [boxes["s"], boxes["o"]], mode=mask_mode),
    }
    question = QUESTION.format(s=subject, r=relation, o=obj)
    prompt = PROMPT_TMPL.format(question=question)
    encoded = [processor(text=prompt, images=views[k], return_tensors="pt") for k in ("orig", "s", "o", "so")]
    pixel_values = torch.cat([e["pixel_values"] for e in encoded], dim=0).to(_device)
    input_ids = encoded[0]["input_ids"].to(_device).expand(pixel_values.shape[0], -1)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, pixel_values=pixel_values).logits
        log_probs = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
        log_odds = (log_probs[:, _yes_id] - log_probs[:, _no_id]).cpu().numpy()
    values = {k: float(v) for k, v in zip(("orig", "s", "o", "so"), log_odds)}
    return {
        "a_orig": values["orig"],
        "d_s": values["orig"] - values["s"],
        "d_o": values["orig"] - values["o"],
        "d_so": values["orig"] - values["so"],
    }


def validate_full_gate(outdir):
    gate_path = Path(outdir) / "pilot_gate.json"
    if not gate_path.exists():
        raise RuntimeError("full scoring is blocked: pilot_gate.json does not exist")
    import json
    with gate_path.open(encoding="utf-8") as f:
        gate = json.load(f)
    if gate.get("status") != "PASS":
        raise RuntimeError(f"full scoring is blocked: pilot status is {gate.get('status')!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=tuple(STAGE_FILES))
    parser.add_argument("--outdir", default=str(BASE))
    parser.add_argument("--pairs", help="override the stage pair CSV")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--mask", default="blur", choices=("gray", "mean", "blur"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    if args.stage == "full":
        validate_full_gate(outdir)
    pair_path = Path(args.pairs) if args.pairs else outdir / STAGE_FILES[args.stage]
    score_path = outdir / "scored_candidates.csv"
    validate_checkpoint_mask(score_path, args.mask)
    pairs = load_pairs(pair_path)
    done = load_done_keys(score_path)
    expected = len(pairs) * len(RELATIONS)
    pending_count = sum(
        candidate_key(row) not in done for row in iter_candidate_rows(pairs)
    )
    print(
        f"[info] stage={args.stage} pairs={len(pairs)} candidates={expected} "
        f"already_scored={expected - pending_count} pending={pending_count}",
        flush=True,
    )
    if not pending_count:
        print("[done] no pending candidates", flush=True)
        return

    model, processor = load_model(args.model_path)
    fresh = not score_path.exists() or score_path.stat().st_size == 0
    outdir.mkdir(parents=True, exist_ok=True)
    failures = 0
    with score_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_FIELDS, extrasaction="ignore")
        if fresh:
            writer.writeheader()
        index = 0
        for row in iter_candidate_rows(pairs):
            if candidate_key(row) in done:
                continue
            index += 1
            started = time.perf_counter()
            try:
                image = Image.open(row["image_path"]).convert("RGB")
                boxes = {"s": parse_box(row["subject_box"]), "o": parse_box(row["object_box"])}
                values = score_candidate(
                    model, processor, image,
                    row["subject_name"], row["candidate_r"], row["object_name"],
                    boxes, mask_mode=args.mask,
                )
                if not all(math.isfinite(v) for v in values.values()):
                    raise ValueError(f"non-finite scores: {values}")
            except Exception as exc:
                failures += 1
                print(f"[error] {row['pair_id']} / {row['candidate_r']}: {exc}", flush=True)
                continue
            record = dict(row)
            record.update(values)
            record["elapsed_sec"] = round(time.perf_counter() - started, 6)
            record["mask"] = args.mask
            writer.writerow(record)
            f.flush()
            done.add(candidate_key(row))
            if index % 10 == 0 or index == pending_count:
                print(
                    f"[progress] {index}/{pending_count} new candidates; failures={failures}",
                    flush=True,
                )
    print(
        f"[done] checkpoint={score_path} newly_scored={pending_count - failures} failures={failures}",
        flush=True,
    )
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
