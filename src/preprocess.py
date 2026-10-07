"""DICOM series -> fixed-size uint8 volume (D, H, W).

The same `load_volume` is used for training and in the Kaggle inference notebook.

    python src/preprocess.py --root E:/Documents/Projects/data/subset
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

DEPTH, SIZE = 32, 256


def read_series(series_dir):
    """Read all slices, sorted along the slice normal. Returns float32 (N, H, W) list-safe."""
    slices = []
    for e in os.scandir(series_dir):
        if not e.name.endswith(".dcm"):
            continue
        try:
            ds = pydicom.dcmread(e.path)
            img = ds.pixel_array.astype(np.float32)
        except Exception:
            continue  # corrupt / partially extracted file
        img = img * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        if hasattr(ds, "ImageOrientationPatient") and hasattr(ds, "ImagePositionPatient"):
            o = np.array(ds.ImageOrientationPatient, dtype=np.float64)
            pos = float(np.dot(np.cross(o[:3], o[3:]), np.array(ds.ImagePositionPatient, dtype=np.float64)))
        else:
            pos = float(getattr(ds, "InstanceNumber", 0))
        slices.append((pos, img))
    if not slices:
        raise ValueError(f"no readable slices in {series_dir}")
    slices.sort(key=lambda s: s[0])
    return [img for _, img in slices]


def pad_square(img):
    h, w = img.shape
    s = max(h, w)
    out = np.zeros((s, s), dtype=img.dtype)
    y, x = (s - h) // 2, (s - w) // 2
    out[y:y + h, x:x + w] = img
    return out


def load_volume(series_dir, depth=DEPTH, size=SIZE):
    imgs = read_series(series_dir)
    # Slices of one series can occasionally differ in shape; resize each to a square first.
    t = torch.stack([F.interpolate(torch.from_numpy(pad_square(i))[None, None], size=(size, size),
                                   mode="bilinear", align_corners=False)[0, 0] for i in imgs])
    t = F.interpolate(t[None, None], size=(depth, size, size), mode="trilinear", align_corners=False)[0, 0]
    v = t.numpy()
    lo, hi = np.percentile(v[v > 0], [0.5, 99.5]) if (v > 0).any() else (0.0, 1.0)
    v = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    return (v * 255).astype(np.uint8), len(imgs)


def _work(job):
    series_dir, out_path = job
    if out_path.exists():
        return out_path.stem, -1, ""
    try:
        vol, n = load_volume(series_dir)
        np.save(out_path, vol)
        return out_path.stem, n, ""
    except Exception as ex:
        return out_path.stem, 0, repr(ex)


def save_montage(vol_dir, series_csv, out_png, n=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = pd.read_csv(series_csv).sample(n, random_state=0)
    fig, axes = plt.subplots(n, 5, figsize=(12, 2.6 * n))
    for r, (_, row) in enumerate(s.iterrows()):
        v = np.load(vol_dir / f"{row.SeriesInstanceUID}.npy")
        for c, z in enumerate(np.linspace(4, DEPTH - 5, 5).astype(int)):
            axes[r, c].imshow(v[z], cmap="gray")
            axes[r, c].axis("off")
        axes[r, 0].set_title(f"{row.Anatomical_Plane} FS={row.Fat_Suppression}", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\Documents\Projects\data\subset")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root)
    series = pd.read_csv(root / f"{args.split}_series.csv")
    vol_dir = root / "volumes"
    vol_dir.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    jobs = [(root / f"{args.split}_series" / r.StudyInstanceUID / r.SeriesInstanceUID,
             vol_dir / f"{r.SeriesInstanceUID}.npy") for r in series.itertuples()]

    results = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_work, jobs, chunksize=4)):
            results.append(res)
            if (i + 1) % 250 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)

    res = pd.DataFrame(results, columns=["SeriesInstanceUID", "n_slices", "error"])
    failed = res[res["n_slices"] == 0]
    print(f"done: {len(res)} series, {len(failed)} failed")
    if len(failed):
        print(failed.head(20).to_string())
    res.to_csv(root / f"preprocess_log_{args.split}.csv", index=False)
    save_montage(vol_dir, root / f"{args.split}_series.csv", root / f"montage_{args.split}.png")


if __name__ == "__main__":
    main()
