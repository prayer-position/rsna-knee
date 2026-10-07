"""Predict test studies with an ensemble of fold checkpoints -> submission.csv.

Works locally and in an offline Kaggle notebook (copy src/ into a Kaggle dataset alongside the .pt files).

    python src/infer.py --data E:/Documents/Projects/data/rsna-knee-abnormality-detection \
        --ckpts runs/base/fold*.pt --out submission.csv
"""
import argparse
import glob
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import TARGETS
from dataset import KneeDataset, pick_series
from model import KneeNet
from preprocess import load_volume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/input/rsna-knee-abnormality-detection")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    data = Path(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = pd.read_csv(data / "test.csv")
    series = pd.read_csv(data / "test_series.csv")
    study_series = pick_series(series)  # no slice counts at test time; any series in the slot is fine
    study_of = dict(zip(series["SeriesInstanceUID"], series["StudyInstanceUID"]))

    @lru_cache(maxsize=16)
    def load_fn(uid):
        return load_volume(data / "test_series" / study_of[uid] / uid)[0]

    ckpts = [p for pattern in args.ckpts for p in sorted(glob.glob(pattern))]
    models = []
    for p in ckpts:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        c = ck["cfg"]
        m = KneeNet(c["backbone"], pretrained=False, n_slices=c["n_slices"], grad_ckpt=False)
        m.load_state_dict(ck["model"])
        models.append(m.to(device).eval())
    print(f"{len(models)} checkpoints on {device}")

    c = ck["cfg"]
    ds = KneeDataset(test["StudyInstanceUID"], study_series, load_fn, n_slices=c["n_slices"], crop=c["crop"])
    dl = DataLoader(ds, batch_size=1, num_workers=0)
    ids, preds = [], []
    with torch.no_grad():
        for batch in dl:
            x, mask = batch["x"].to(device), batch["mask"].to(device)
            if not mask.any():
                preds.append(np.full((1, len(TARGETS)), 0.5))
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    ps = [torch.sigmoid(m(x, mask).float()).cpu().numpy() for m in models]
                preds.append(np.mean(ps, axis=0))
            ids += list(batch["id"])

    sub = pd.DataFrame(np.concatenate(preds), columns=TARGETS)
    sub.insert(0, "StudyInstanceUID", ids)
    sub.to_csv(args.out, index=False)
    print(sub.head())


if __name__ == "__main__":
    main()
