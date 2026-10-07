"""Study-level dataset: up to one series per view slot, each a (S, 3, H, W) 2.5D stack."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common import SLOTS, series_slot


def pick_series(series_df, n_slices=None):
    """Map StudyInstanceUID -> {slot: SeriesInstanceUID}, keeping the longest series per slot."""
    df = series_df.copy()
    df["slot"] = df.apply(series_slot, axis=1)
    df["n"] = df["SeriesInstanceUID"].map(n_slices).fillna(0) if n_slices is not None else 0
    df = df.sort_values("n", ascending=False).drop_duplicates(["StudyInstanceUID", "slot"])
    return {sid: dict(zip(g["slot"], g["SeriesInstanceUID"])) for sid, g in df.groupby("StudyInstanceUID")}


def to_stack(vol, n_slices):
    """(D, H, W) uint8 -> (n_slices, 3, H, W): evenly spaced centre slices with their neighbours."""
    d = vol.shape[0]
    centres = np.linspace(1, d - 2, n_slices).round().astype(int)
    return np.stack([vol[c - 1:c + 2] for c in centres])


class KneeDataset(Dataset):
    def __init__(self, study_ids, study_series, load_fn, labels=None, weights=None,
                 n_slices=16, crop=224, train=False):
        self.ids = list(study_ids)
        self.study_series = study_series
        self.load_fn = load_fn  # series_uid -> (D, H, W) uint8 volume
        self.labels, self.weights = labels, weights
        self.n_slices, self.crop, self.train = n_slices, crop, train

    def __len__(self):
        return len(self.ids)

    def _augment(self, x):
        # x: (S, 3, H, W) float in [0, 1]. No flips: they would swap medial/lateral anatomy.
        _, _, h, w = x.shape
        y0, x0 = np.random.randint(0, h - self.crop + 1), np.random.randint(0, w - self.crop + 1)
        x = x[..., y0:y0 + self.crop, x0:x0 + self.crop]
        gamma = np.exp(np.random.uniform(-0.25, 0.25))
        x = x.clamp(min=0) ** gamma
        return x * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05)

    def _centre(self, x):
        h, w = x.shape[-2:]
        y0, x0 = (h - self.crop) // 2, (w - self.crop) // 2
        return x[..., y0:y0 + self.crop, x0:x0 + self.crop]

    def __getitem__(self, i):
        sid = self.ids[i]
        slots = self.study_series.get(sid, {})
        x = torch.zeros(len(SLOTS), self.n_slices, 3, self.crop, self.crop)
        mask = torch.zeros(len(SLOTS), dtype=torch.bool)
        for k, slot in enumerate(SLOTS):
            if slot not in slots:
                continue
            try:
                vol = self.load_fn(slots[slot])
            except Exception:
                continue
            s = torch.from_numpy(to_stack(vol, self.n_slices)).float() / 255.0
            x[k] = self._augment(s) if self.train else self._centre(s)
            mask[k] = True
        out = {"x": (x - 0.5) / 0.25, "mask": mask, "id": sid}
        if self.labels is not None:
            y = self.labels.loc[sid].to_numpy(dtype=np.float32)
            out["y"] = torch.from_numpy(np.nan_to_num(y, nan=0.0))
            out["y_mask"] = torch.from_numpy(~np.isnan(y))
            out["w"] = torch.tensor(float(self.weights.loc[sid]))
        return out


class npy_loader:
    """Picklable (Windows DataLoader workers use spawn) series_uid -> volume loader."""

    def __init__(self, vol_dir):
        self.vol_dir = str(vol_dir)

    def __call__(self, uid):
        return np.load(f"{self.vol_dir}/{uid}.npy")


def load_n_slices(log_csv):
    try:
        log = pd.read_csv(log_csv)
        return log.set_index("SeriesInstanceUID")["n_slices"]
    except FileNotFoundError:
        return None
