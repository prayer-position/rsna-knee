"""Build a stratified development subset of the RSNA knee dataset.

Keeps every gold-labeled study, adds unlabeled studies stratified by report
language and series count, hard-links the DICOMs into a Kaggle-style layout
and writes 5 folds.

    python src/make_subset.py --n 500
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from langdetect import DetectorFactory, detect
from sklearn.model_selection import StratifiedKFold

from common import TARGETS, series_slot

DetectorFactory.seed = 0


def detect_lang(text):
    try:
        return detect(str(text)[:400])
    except Exception:
        return "unk"


def study_is_complete(root, study, series_ids):
    """True if every series folder exists and holds non-empty .dcm files."""
    for se in series_ids:
        d = root / study / se
        if not d.is_dir():
            return False
        files = [e for e in os.scandir(d) if e.name.endswith(".dcm")]
        if not files or any(e.stat().st_size == 0 for e in files):
            return False
    return True


def allocate(sizes, total):
    """Largest-remainder proportional allocation of `total` over strata."""
    raw = sizes / sizes.sum() * total
    alloc = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - alloc))[: total - alloc.sum()]:
        alloc.iloc[i] += 1
    return np.minimum(alloc, sizes)


def link_study(src_root, dst_root, study, series_ids):
    for se in series_ids:
        src, dst = src_root / study / se, dst_root / study / se
        dst.mkdir(parents=True, exist_ok=True)
        for e in os.scandir(src):
            target = dst / e.name
            if not target.exists():
                os.link(e.path, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:\Documents\Projects\data\rsna-knee-abnormality-detection")
    ap.add_argument("--out", default=r"E:\Documents\Projects\data\subset")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    rng = np.random.default_rng(args.seed)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    series_by_study = series.groupby("StudyInstanceUID")["SeriesInstanceUID"].apply(list)

    train["gold"] = train[TARGETS].notna().all(axis=1)
    train["lang"] = train["Report"].map(detect_lang)
    counts = train["lang"].value_counts()
    train["lang_b"] = train["lang"].where(train["lang"].map(counts) >= 30, "other")
    n_series = train["StudyInstanceUID"].map(series_by_study.str.len())
    train["nser_b"] = pd.cut(n_series, [0, 4, 5, 6, 99], labels=["<=4", "5", "6", ">=7"]).astype(str)
    train["stratum"] = train["lang_b"] + "|" + train["nser_b"]
    print("Report languages:\n", counts.to_string())

    root = data / "train_series"
    selected, skipped = [], []
    for sid in train.loc[train["gold"], "StudyInstanceUID"]:
        (selected if study_is_complete(root, sid, series_by_study[sid]) else skipped).append(sid)
    print(f"gold studies usable: {len(selected)} (skipped {len(skipped)})")

    pool = train[~train["gold"]]
    sizes = pool["stratum"].value_counts()
    alloc = allocate(sizes, args.n - len(selected))
    for stratum, k in alloc.items():
        cand = pool.loc[pool["stratum"] == stratum, "StudyInstanceUID"].to_numpy()
        taken = 0
        for sid in rng.permutation(cand):
            if taken == k:
                break
            if study_is_complete(root, sid, series_by_study[sid]):
                selected.append(sid)
                taken += 1
            else:
                skipped.append(sid)
    print(f"selected {len(selected)} studies, skipped {len(skipped)} incomplete")

    sub = train[train["StudyInstanceUID"].isin(selected)].reset_index(drop=True)
    sub_series = series[series["StudyInstanceUID"].isin(selected)].reset_index(drop=True)

    # Folds: gold stratified by positive count, the rest by stratum.
    sub["fold"] = -1
    for mask, key in [(sub["gold"], sub[TARGETS].sum(axis=1).clip(upper=4).astype(str)),
                      (~sub["gold"], sub["lang_b"])]:
        idx = sub.index[mask]
        y = key[mask].to_numpy()
        # Merge classes too rare to split.
        vc = pd.Series(y).value_counts()
        y = np.where(pd.Series(y).map(vc) < args.folds, "rare", y)
        skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
        for f, (_, te) in enumerate(skf.split(idx, y)):
            sub.loc[idx[te], "fold"] = f

    out.mkdir(parents=True, exist_ok=True)
    sub[["StudyInstanceUID", "Report"] + TARGETS].to_csv(out / "train.csv", index=False)
    sub_series.to_csv(out / "train_series.csv", index=False)
    sub[["StudyInstanceUID", "fold", "gold", "lang", "stratum"]].to_csv(out / "folds.csv", index=False)
    pd.Series(skipped, name="StudyInstanceUID").to_csv(out / "skipped_incomplete.csv", index=False)

    for i, sid in enumerate(sub["StudyInstanceUID"]):
        link_study(root, out / "train_series", sid, series_by_study[sid])
        if (i + 1) % 100 == 0:
            print(f"linked {i + 1}/{len(sub)}")

    slots = sub_series.apply(series_slot, axis=1)
    print("\nSubset summary")
    print("studies:", len(sub), "| gold:", int(sub["gold"].sum()), "| series:", len(sub_series))
    print("per fold:\n", sub.groupby("fold")["gold"].agg(["size", "sum"]).to_string())
    print("series slots:\n", slots.value_counts().to_string())
    print("languages:\n", sub["lang"].value_counts().to_string())


if __name__ == "__main__":
    main()
