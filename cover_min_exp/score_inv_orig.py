#!/usr/bin/env python
"""Score the inversion question's raw Yes/No log-odds for eval R-Bench claims.

The REPAIR output-flip measurement: after the decision layer REPAIRs a claim to
the inverse relation (o, r^-1, s), this is the model's answer to the repaired
question. One forward pass per claim (original image only).

Writes a_inv_orig.csv with key (image_id, s, r, o) -> a_inv_orig.
"""
import csv
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cover_rbench as cr
from PIL import Image

BASE = "/root/autodl-tmp/cover_min_exp"
model, processor = cr.load_model()
yes_id = cr._yes_id
no_id = cr._no_id

rows = list(csv.DictReader(open(BASE + "/scored_rbench_full.csv")))
# r_inv lives in the claims file, not the scored file — join by (image_id, s, r, o)
rinv_map = {}
for c in csv.DictReader(open(BASE + "/rbench_claims_full.csv")):
    rinv_map[(c["image_id"], c["s"], c["r"], c["o"])] = (c.get("r_inv") or "").strip()
for r in rows:
    r["_rinv"] = rinv_map.get((r["image_id"], r["s"], r["r"], r["o"]), "")
out_path = BASE + "/a_inv_orig.csv"
done = set()
fresh = not os.path.exists(out_path)
f = open(out_path, "a", newline="")
w = csv.writer(f)
if fresh:
    w.writerow(["image_id", "s", "r", "o", "a_inv_orig"])
if not fresh:
    for r in csv.DictReader(open(out_path)):
        done.add((r["image_id"], r["s"], r["r"], r["o"]))

n_eval = sum(1 for r in rows if r["split"] == "eval")
count = 0
for i, r in enumerate(rows):
    if r["split"] != "eval":
        continue
    r_inv = r["_rinv"]
    if not r_inv:
        continue
    key = (r["image_id"], r["s"], r["r"], r["o"])
    if key in done:
        continue
    img = Image.open(f"{BASE}/../data/R-bench/validation/{r['image_id']}").convert("RGB")
    q = f"Is it true that the {r['o']} is {r_inv} the {r['s']} in the image?"
    with torch.inference_mode():
        enc = processor(text=cr.PROMPT_TMPL.format(q=q), images=img, return_tensors="pt")
        input_ids = enc["input_ids"].to(cr._device)
        pixel_values = enc["pixel_values"].to(cr._device)
        logits = model(input_ids=input_ids, pixel_values=pixel_values).logits
        lp = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
        a = float((lp[:, yes_id] - lp[:, no_id]).cpu().numpy()[0])
    w.writerow([r["image_id"], r["s"], r["r"], r["o"], round(a, 6)])
    f.flush()
    done.add(key)
    count += 1
    if count % 50 == 0:
        print(f"[progress] {count} scored (of ~{n_eval} eval claims)", flush=True)
f.close()
print(f"[done] {count} inversion answers scored", flush=True)
