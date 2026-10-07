"""Train KneeNet with k-fold CV; reports macro AUC on gold-labeled validation studies.

    python src/train.py --config configs/base.yaml
    python src/train.py --config configs/base.yaml --set epochs=1 limit_studies=20 folds=[0]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from common import TARGETS
from dataset import KneeDataset, load_n_slices, npy_loader, pick_series
from labels import get_labels
from model import KneeNet


def macro_auc(y, p):
    aucs = {}
    for j, t in enumerate(TARGETS):
        if len(np.unique(y[:, j])) == 2:
            aucs[t] = roc_auc_score(y[:, j], p[:, j])
    return (float(np.mean(list(aucs.values()))) if aucs else float("nan")), aucs


def load_config():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. epochs=1 folds=[0]")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ids, preds = [], []
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(batch["x"].to(device, non_blocking=True), batch["mask"].to(device))
        preds.append(torch.sigmoid(logits.float()).cpu().numpy())
        ids += list(batch["id"])
    return ids, np.concatenate(preds)


def train_fold(cfg, fold, folds, labels, weights, study_series, load_fn, device):
    tr_ids = folds.loc[(folds["fold"] != fold) & (folds["StudyInstanceUID"].map(weights) > 0), "StudyInstanceUID"]
    va_ids = folds.loc[(folds["fold"] == fold) & folds["gold"], "StudyInstanceUID"]
    if cfg["limit_studies"]:
        tr_ids, va_ids = tr_ids[: cfg["limit_studies"]], va_ids[: max(4, cfg["limit_studies"] // 4)]
    print(f"[fold {fold}] train {len(tr_ids)} studies, val {len(va_ids)} gold studies")

    kw = dict(study_series=study_series, load_fn=load_fn, labels=labels, weights=weights,
              n_slices=cfg["n_slices"], crop=cfg["crop"])
    dl_kw = dict(num_workers=cfg["num_workers"], pin_memory=True, persistent_workers=cfg["num_workers"] > 0)
    tr_dl = DataLoader(KneeDataset(tr_ids, train=True, **kw), batch_size=cfg["batch_size"],
                       shuffle=True, drop_last=True, **dl_kw)
    va_dl = DataLoader(KneeDataset(va_ids, **kw), batch_size=cfg["batch_size"], **dl_kw)

    ckpt = Path(cfg["out_dir"]) / f"fold{fold}.pt"
    if cfg.get("resume") and ckpt.exists():
        # Fold already trained: only recompute its out-of-fold predictions.
        model = KneeNet(cfg["backbone"], False, cfg["n_slices"], cfg["drop"], grad_ckpt=False).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
        ids, p = predict(model, va_dl, device)
        print(f"[fold {fold}] resumed from {ckpt.name}: val gold macro-AUC "
              f"{macro_auc(labels.loc[ids].to_numpy(), p)[0]:.4f}", flush=True)
        return pd.DataFrame(p, index=ids, columns=TARGETS)

    model = KneeNet(cfg["backbone"], cfg["pretrained"], cfg["n_slices"], cfg["drop"], cfg["grad_ckpt"]).to(device)
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": cfg["backbone_lr"]}, {"params": rest, "lr": cfg["lr"]}],
                            weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * max(1, len(tr_dl) // cfg["accum_steps"])
    warm = cfg["warmup_epochs"] * max(1, len(tr_dl) // cfg["accum_steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    out = Path(cfg["out_dir"])
    best, best_preds = -1.0, None
    for epoch in range(cfg["epochs"]):
        model.train()
        t0, losses = time.time(), []
        opt.zero_grad()
        for i, batch in enumerate(tr_dl):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(batch["x"].to(device, non_blocking=True), batch["mask"].to(device))
            y, ym, w = batch["y"].to(device), batch["y_mask"].to(device), batch["w"].to(device)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
            loss = (loss * ym * w[:, None]).sum() / (ym * w[:, None]).sum().clamp(min=1)
            (loss / cfg["accum_steps"]).backward()
            if (i + 1) % cfg["accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            losses.append(loss.item())

        ids, p = predict(model, va_dl, device)
        auc, _ = macro_auc(labels.loc[ids].to_numpy(), p)
        mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
        print(f"[fold {fold}] epoch {epoch} loss {np.mean(losses):.4f} val gold macro-AUC {auc:.4f} "
              f"({time.time() - t0:.0f}s, peak {mem:.1f} GB)", flush=True)
        if np.isnan(auc) or auc > best:
            best, best_preds = auc, pd.DataFrame(p, index=ids, columns=TARGETS)
            torch.save({"model": model.state_dict(), "cfg": cfg}, out / f"fold{fold}.pt")
    return best_preds


def main():
    cfg = load_config()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")

    root = Path(cfg["data_root"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"))

    labels, weights = get_labels(root / "train.csv", cfg["weak_labels"], cfg["weak_weight"])
    folds = pd.read_csv(root / "folds.csv")
    study_series = pick_series(pd.read_csv(root / "train_series.csv"), load_n_slices(root / "preprocess_log_train.csv"))
    load_fn = npy_loader(root / "volumes")

    oof = pd.concat([train_fold(cfg, f, folds, labels, weights, study_series, load_fn, device) for f in cfg["folds"]])
    oof.to_csv(out / "oof.csv", index_label="StudyInstanceUID")
    auc, per_target = macro_auc(labels.loc[oof.index].to_numpy(), oof.to_numpy())
    print(f"\nOOF gold macro-AUC: {auc:.4f}")
    for t, a in per_target.items():
        print(f"  {t:18s} {a:.3f}")
    json.dump({"oof_macro_auc": auc, "per_target": per_target}, open(out / "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
