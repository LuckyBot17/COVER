#!/usr/bin/env python
"""Mask-type ablation for the COVER minimal experiment.

Re-scores a sample of claims under three matched negative-control variants
(gray block, region mean-fill, heavy blur) for the identity view only, and
reports the per-variant operator signal (median drop for true vs false claims,
AUROC of the evidence drop).
"""
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

MODEL_PATH = "/root/autodl-tmp/llava-1.5-7b/master"
PROMPT_TMPL = "USER: <image>\n{q}\nASSISTANT:"
QUESTION = "Is it true that the {s} is {r} the {o} in the image?"
VARIANTS = ["gray", "mean", "blur"]


def main():
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    claims = list(csv.DictReader(open("/root/autodl-tmp/cover_min_exp/claims.csv")))
    scored = list(csv.DictReader(open("/root/autodl-tmp/cover_min_exp/scored_claims.csv")))
    a_orig = { (r["image_id"], r["s"], r["r"], r["o"]): float(r["a_orig"]) for r in scored }
    # sample confident claims to probe the informative regime
    rng = random.Random(7)
    c = [r for r in claims if (r["image_id"], r["s"], r["r"], r["o"]) in a_orig]
    c.sort(key=lambda r: a_orig[(r["image_id"], r["s"], r["r"], r["o"])], reverse=True)
    true_samp = [r for r in c if r["label"] == "1"][:20]
    false_samp = [r for r in c if r["label"] == "0"][:20]
    sample = true_samp + false_samp
    rng.shuffle(sample)
    print(f"[info] ablation sample: {len(sample)} claims "
          f"(true {sum(1 for r in sample if r['label']=='1')})", flush=True)

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True).eval().to("cuda")
    proc = AutoProcessor.from_pretrained(MODEL_PATH)
    tok = proc.tokenizer
    yes_id = tok("Yes", add_special_tokens=False)["input_ids"][0]
    no_id = tok("No", add_special_tokens=False)["input_ids"][0]

    def parse_box(s):
        return tuple(int(v) for v in s.split("|"))

    def log_odds(q, img):
        enc = proc(text=PROMPT_TMPL.format(q=q), images=img, return_tensors="pt")
        ids = enc["input_ids"].to("cuda")
        px = enc["pixel_values"].to("cuda")
        with torch.inference_mode():
            out = model(input_ids=ids, pixel_values=px)
        lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        return float(lp[yes_id] - lp[no_id])

    def variants_of(img, sb, ob):
        W, H = img.size
        res = {}
        for name in VARIANTS:
            m = img.copy()
            d = ImageDraw.Draw(m)
            for (x, y, w, h) in (sb, ob):
                if name == "gray":
                    d.rectangle([x, y, x + w, y + h], fill=(128, 128, 128))
                elif name == "mean":
                    crop = img.crop((x, y, x + w, y + h))
                    arr = np.asarray(crop, dtype=np.float32).reshape(-1, 3).mean(0).round().astype(int)
                    d.rectangle([x, y, x + w, y + h], fill=tuple(arr))
                else:  # blur: strongly blur the region, tile blurred patch
                    patch = img.crop((max(0, x - 20), max(0, y - 20),
                                      min(W, x + w + 20), min(H, y + h + 20)))
                    patch = patch.filter(ImageFilter.GaussianBlur(radius=max(8, int(w * 0.15))))
                    m.paste(patch, (max(0, x - 20), max(0, y - 20)))
            res[name] = m
        return res

    rows = []
    for r in sample:
        img = Image.open(r["image_path"]).convert("RGB")
        sb, ob = parse_box(r["s_box"]), parse_box(r["o_box"])
        q = QUESTION.format(s=r["s"], r=r["r"], o=r["o"])
        a_pos = log_odds(q, img)
        rows.append({"image_id": r["image_id"], "s": r["s"], "r": r["r"], "o": r["o"],
                     "label": int(r["label"]), "a_orig": a_pos})
        for name, m in variants_of(img, sb, ob).items():
            rows[-1][name] = a_pos - log_odds(q, m)
        if len(rows) % 5 == 0:
            print(f"[progress] {len(rows)}/{len(sample)}", flush=True)

    y = np.array([r["label"] for r in rows])
    print("\n=== mask-type ablation (identity view, both-entity mask) ===")
    for name in VARIANTS:
        dv = np.array([r[name] for r in rows])
        dt, df = dv[y == 1], dv[y == 0]
        # AUROC of drop (true should drop more)
        from scipy.stats import rankdata
        rk = rankdata(dv)
        n1, n0 = int(y.sum()), int((1 - y).sum())
        u = rk[y == 1].sum() - n1 * (n1 + 1) / 2
        print(f"  {name:6s}: median_drop true={np.median(dt):+.3f} false={np.median(df):+.3f} "
              f"g={np.median(dt) - np.median(df):+.3f} | AUROC(d)={u / (n1 * n0):.3f} | "
              f"mean true/false = {dt.mean():+.3f}/{df.mean():+.3f}", flush=True)

    with open("/root/autodl-tmp/cover_min_exp/ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "s", "r", "o", "label", "a_orig"] + VARIANTS)
        w.writeheader()
        w.writerows(rows)
    print("[done] wrote ablation.csv", flush=True)


if __name__ == "__main__":
    main()
