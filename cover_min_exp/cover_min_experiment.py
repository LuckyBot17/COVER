#!/usr/bin/env python
"""COVER minimal falsifiable experiment on GQA-val claims with LLaVA-1.5-7B.

For each claim (s, r, o) with subject/object boxes:
  - positive controls: identity claim + argument-inversion claim
  - negative controls per view: subject masked, object masked, both masked
  - scores: Yes/No log-odds -> evidence differences d_{e,k}
After all claims are scored, calibration statistics b_j/g_j are fit on the
calibration split, then standardized measurements x_j, common support mu and
overidentification conflict J are computed on the evaluation split, along with
AUROC comparisons against the raw confidence score and the within-confidence-
stratum test that is the gate of the whole method.

Resumable: scored claims are checkpointed to a CSV; rerun skips them.
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

MODEL_PATH = "/root/autodl-tmp/llava-1.5-7b/master"
QUESTION = "Is it true that the {s} is {r} the {o} in the image?"
PROMPT_TMPL = "USER: <image>\n{q}\nASSISTANT:"
MASK_COLOR = (128, 128, 128)
OPS = [("id", "s"), ("id", "o"), ("id", "so"), ("inv", "s"), ("inv", "o"), ("inv", "so")]

_device = None
_model = None
_processor = None
_yes_id = None
_no_id = None


def load_model():
    global _model, _processor, _yes_id, _no_id, _device
    if _model is not None:
        return _model, _processor
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    _device = "cuda" if torch_available() else "cpu"
    torch = __import__("torch")
    dtype = torch.float16
    _model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    _model = _model.eval().to(_device)
    _processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tok = _processor.tokenizer
    for tok_str in ("Yes", "No"):
        ids = tok(tok_str, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            print(f"[warn] token '{tok_str}' is {len(ids)} tokens: {ids}", flush=True)
    _yes_id = tok("Yes", add_special_tokens=False)["input_ids"][0]
    _no_id = tok("No", add_special_tokens=False)["input_ids"][0]
    print(f"[info] model loaded on {_device}; yes_id={_yes_id} no_id={_no_id}", flush=True)
    return _model, _processor


def torch_available():
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def mask_image(img, boxes, mode="blur"):
    """Render a matched negative control. mode: gray | mean | blur."""
    img = img.copy()
    if mode == "gray":
        draw = ImageDraw.Draw(img)
        for (x, y, w, h) in boxes:
            draw.rectangle([x, y, x + w, y + h], fill=MASK_COLOR)
    elif mode == "mean":
        draw = ImageDraw.Draw(img)
        for (x, y, w, h) in boxes:
            crop = img.crop((x, y, x + w, y + h))
            arr = np.asarray(crop, dtype=np.float32).reshape(-1, 3).mean(0).round().astype(int)
            draw.rectangle([x, y, x + w, y + h], fill=tuple(arr))
    else:  # blur
        W, H = img.size
        for (x, y, w, h) in boxes:
            cx0, cy0 = max(0, x - 20), max(0, y - 20)
            cx1, cy1 = min(W, x + w + 20), min(H, y + h + 20)
            patch = img.crop((cx0, cy0, cx1, cy1)).filter(
                ImageFilter.GaussianBlur(radius=max(8, int(w * 0.15))))
            img.paste(patch, (cx0, cy0))
    return img


def parse_box(s):
    return tuple(int(v) for v in s.split("|"))


def load_image(path):
    img = Image.open(path).convert("RGB")
    return img


def score_claim(model, processor, img, s, r, o, boxes, r_inv=None, mask_mode="blur"):
    """Return dict of log-odds for views {id, inv} x {orig, s, o, so}."""
    import torch

    masked = {
        "s": mask_image(img, [boxes["s"]], mode=mask_mode),
        "o": mask_image(img, [boxes["o"]], mode=mask_mode),
        "so": mask_image(img, [boxes["s"], boxes["o"]], mode=mask_mode),
    }
    images = {"orig": img, **masked}
    prompts = {
        "id": PROMPT_TMPL.format(q=QUESTION.format(s=s, r=r, o=o)),
        "inv": PROMPT_TMPL.format(q=QUESTION.format(s=o, r=r_inv, o=s)),
    }
    out = {}
    with torch.inference_mode():
        for vname, q in prompts.items():
            # batch the 4 images of this view in one forward (same canvas size)
            pix = []
            for k in ("orig", "s", "o", "so"):
                enc = processor(text=q, images=images[k], return_tensors="pt")
                pix.append(enc["pixel_values"])
            pixel_values = torch.cat(pix, dim=0).to(_device)
            input_ids = processor(text=q, images=images["orig"], return_tensors="pt")["input_ids"].to(_device)
            input_ids = input_ids.expand(pixel_values.shape[0], -1)
            logits = model(input_ids=input_ids, pixel_values=pixel_values).logits
            lp = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
            lo = (lp[:, _yes_id] - lp[:, _no_id]).cpu().numpy()
            out[vname] = {k: float(v) for k, v in zip(("orig", "s", "o", "so"), lo)}
    return out


def evidence_diffs(scores):
    d = {}
    for vname in ("id", "inv"):
        for k in ("s", "o", "so"):
            d[f"{vname}:{k}"] = scores[vname]["orig"] - scores[vname][k]
    return d


def auroc(y, s):
    try:
        from scipy.stats import rankdata

        y = np.asarray(y, dtype=float)
        s = np.asarray(s, dtype=float)
        mask = ~np.isnan(s)
        y, s = y[mask], s[mask]
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        ranks = rankdata(s, method="average")
        u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
        return float(u / (n_pos * n_neg))
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default="/root/autodl-tmp/cover_min_exp/claims.csv")
    ap.add_argument("--outdir", default="/root/autodl-tmp/cover_min_exp")
    ap.add_argument("--g-min", type=float, default=0.25)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--n-strata", type=int, default=3)
    ap.add_argument("--mask", default="blur", choices=["gray", "mean", "blur"])
    args = ap.parse_args()

    import torch

    claims = list(csv.DictReader(open(args.claims)))
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    score_path = Path(args.outdir) / f"scored_claims_{args.mask}.csv"
    metrics_path = Path(args.outdir) / f"metrics_{args.mask}.json"
    eval_path = Path(args.outdir) / f"eval_claims_{args.mask}.csv"
    done = set()
    if score_path.exists():
        with open(score_path) as f:
            for row in csv.DictReader(f):
                done.add((row["image_id"], row["s"], row["r"], row["o"]))
    print(f"[info] {len(claims)} claims, {len(done)} already scored", flush=True)

    model, processor = load_model()

    # Score (resumable)
    fieldnames = ["image_id", "s", "o", "r", "r_inv", "label", "split"] + [f"d_{v}:{k}" for v in ("id", "inv") for k in ("s", "o", "so")] + ["a_orig"]
    fresh = score_path.exists() is False
    f = open(score_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    if fresh:
        writer.writeheader()
    for i, row in enumerate(claims):
        key = (row["image_id"], row["s"], row["r"], row["o"])
        if key in done:
            continue
        img = load_image(row["image_path"])
        boxes = {"s": parse_box(row["s_box"]), "o": parse_box(row["o_box"])}
        try:
            scores = score_claim(model, processor, img, row["s"], row["r"], row["o"], boxes, r_inv=row["r_inv"], mask_mode=args.mask)
        except Exception as e:
            print(f"[error] claim {i}: {e}", flush=True)
            continue
        d = evidence_diffs(scores)
        rec = {
            "image_id": row["image_id"], "s": row["s"], "o": row["o"],
            "r": row["r"], "r_inv": row["r_inv"], "label": row["label"],
            "split": row["split"], "a_orig": scores["id"]["orig"],
        }
        rec.update({f"d_{k}": v for k, v in d.items()})
        writer.writerow(rec)
        f.flush()
        done.add(key)
        if (i + 1) % 10 == 0:
            print(f"[progress] {i + 1}/{len(claims)} (newly scored this run: {len(done)})", flush=True)
    f.close()

    # Load all scored rows
    rows = list(csv.DictReader(open(score_path)))
    if not rows:
        print("no scored claims; abort")
        sys.exit(1)

    def getnum(r, k):
        try:
            return float(r[k])
        except (KeyError, ValueError):
            return float("nan")

    for r in rows:
        r["label"] = int(float(r["label"]))
        r["a_orig"] = getnum(r, "a_orig")
        for v in ("id", "inv"):
            for k in ("s", "o", "so"):
                r[f"d_{v}:{k}"] = getnum(r, f"d_{v}:{k}")

    # --- Calibration statistics (b_j, g_j) from the calibration split ---
    calib = [r for r in rows if r["split"] == "calib"]
    bj, gj = {}, {}
    for j in OPS:
        key = f"d_{j[0]}:{j[1]}"
        d_false = [r[key] for r in calib if r["label"] == 0 and not math.isnan(r[key])]
        d_true = [r[key] for r in calib if r["label"] == 1 and not math.isnan(r[key])]
        if not d_false or not d_true:
            continue
        b = float(np.median(d_false))
        g = float(np.median(d_true)) - b
        if g > args.g_min:
            bj[key] = b
            gj[key] = g
    print(f"[info] kept operators: {sorted(gj.keys())}", flush=True)
    print(f"[info] calibration b: { {k: round(v,3) for k,v in bj.items()} }", flush=True)
    print(f"[info] calibration g: { {k: round(v,3) for k,v in gj.items()} }", flush=True)

    # --- Standardized measurements on the evaluation split ---
    def standardized(r):
        xs = []
        raw = []
        for j in OPS:
            key = f"d_{j[0]}:{j[1]}"
            if key in gj:
                d = r[key]
                if not math.isnan(d):
                    xs.append((d - bj[key]) / (gj[key] + 1e-6))
                    raw.append(d)
        if not xs:
            return None
        mu = float(np.mean(xs))
        m = len(xs)
        jstat = float(np.sum((np.array(xs) - mu) ** 2) / max(1, m - 1))
        return {"mu": mu, "J": jstat, "m": m, "mean_drop": float(np.mean(raw))}

    eval_rows = []
    for r in rows:
        if r["split"] != "eval":
            continue
        st = standardized(r)
        if st is None:
            continue
        r["mu"] = st["mu"]
        r["J"] = st["J"]
        r["m"] = st["m"]
        r["mean_drop"] = st["mean_drop"]
        r["A"] = -st["mu"] + args.lam * st["J"]
        eval_rows.append(r)

    print(f"[info] eval claims: {len(eval_rows)} "
          f"(true {sum(1 for r in eval_rows if r['label']==1)}, "
          f"false {sum(1 for r in eval_rows if r['label']==0)})", flush=True)

    if len(eval_rows) < 10:
        print("too few eval claims", flush=True)
        sys.exit(1)

    y = np.array([r["label"] for r in eval_rows])
    scorers = {
        "a_orig (raw confidence)": np.array([r["a_orig"] for r in eval_rows]),
        "mean evidence drop": np.array([r["mean_drop"] for r in eval_rows]),
        "mu (common support)": np.array([r["mu"] for r in eval_rows]),
        "J (overid conflict)": np.array([r["J"] for r in eval_rows]),
        "A = -mu + lam*J": np.array([r["A"] for r in eval_rows]),
    }
    auc = {name: auroc(y, s) for name, s in scorers.items()}
    print("\n=== Overall AUROC on evaluation split ===", flush=True)
    for name, a in auc.items():
        print(f"  {name:32s} {a:.3f}", flush=True)

    # --- Gate test: separation within strata of the raw confidence ---
    a_orig = scorers["a_orig (raw confidence)"]
    qs = [100.0 * k / args.n_strata for k in range(1, args.n_strata)]
    thr = np.percentile(a_orig, qs)
    print("\n=== Gate test: within raw-confidence strata ===", flush=True)
    strata_report = []
    bounds = [-np.inf] + list(thr) + [np.inf]
    for k in range(args.n_strata):
        lo, hi = bounds[k], bounds[k + 1]
        sel = (a_orig > lo) & (a_orig <= hi)
        ys = y[sel]
        if ys.size == 0:
            continue
        mu = scorers["mu (common support)"][sel]
        A = scorers["A = -mu + lam*J"][sel]
        info = {
            "stratum": k,
            "range": (round(float(lo), 2), round(float(hi), 2)),
            "n": int(ys.size),
            "n_true": int(ys.sum()),
            "auc_mu": auroc(ys, mu),
            "auc_A": auroc(ys, A),
            "mean_mu_true": float(np.nanmean(mu[ys == 1])),
            "mean_mu_false": float(np.nanmean(mu[ys == 0])),
        }
        strata_report.append(info)
        print(f"  stratum {k}  a_orig in {info['range']}  n={info['n']} "
              f"(true={info['n_true']})  AUC(mu)={info['auc_mu']:.3f}  "
              f"AUC(A)={info['auc_A']:.3f}  mean_mu true/false = "
              f"{info['mean_mu_true']:.2f}/{info['mean_mu_false']:.2f}", flush=True)

    results = {
        "claims_scored": len(rows),
        "eval_claims": len(eval_rows),
        "operators_kept": list(gj.keys()),
        "calibration": {"b": bj, "g": gj},
        "overall_auroc": auc,
        "strata": strata_report,
    }
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(eval_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "s", "o", "r", "label", "a_orig", "mean_drop", "mu", "J", "m", "A"])
        for r in eval_rows:
            w.writerow([r["image_id"], r["s"], r["o"], r["r"], r["label"],
                        r["a_orig"], r["mean_drop"], r["mu"], r["J"], r["m"], r["A"]])
    print(f"\n[done] {metrics_path.name} and {eval_path.name} written", flush=True)


if __name__ == "__main__":
    main()
