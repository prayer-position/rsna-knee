# RSNA Knee Abnormality Detection

Pipeline for the RSNA Knee Abnormality Detection Kaggle competition. The task: given a knee MRI study, predict the
probability of 12 findings. The model is a **2.5D multi-view attention-MIL network**. A pretrained 2D CNN encodes
individual slices, attention pools slices into series, and a second, per-finding attention pools series into a
study-level prediction for each finding.

For the story so far (data findings, decisions, results, lessons and next steps), see the
[project manifesto](docs/MANIFESTO.md).

- [Status](#status)
- [Quickstart](#quickstart)
- [Repository layout](#repository-layout)
- [The data](#the-data)
- [Pipeline](#pipeline)
- [The model](#the-model)
- [Training](#training)
- [Inference](#inference)
- [Configuration reference](#configuration-reference)
- [Environment](#environment)
- [Known issues and next steps](#known-issues-and-next-steps)
- [References](#references)

---

## Status

| Step | State |
|---|---|
| Data download and extraction | ✅ done (`data/extract-log.txt`) |
| 500-study dev subset ([`make_subset.py`](src/make_subset.py)) | ✅ done: 500 studies, 58 gold, 0 incomplete skipped |
| Preprocessing to `.npy` volumes ([`preprocess.py`](src/preprocess.py)) | ✅ done: 2,778 series, 0 failures |
| Smoke test (1 fold, 2 epochs, 20 studies) | ✅ runs end to end, including inference (`runs/smoke`, not in git) |
| Full 5-fold base run, gold labels only (`runs/base`, not in git) | ✅ done 2026-10-07: OOF macro-AUC **0.554** |
| Weak labels from radiology reports ([`weak_labels.py`](src/weak_labels.py)) | 🔄 pilot on the 58 gold reports: Gemma 4 12B agrees at AUC 0.851; Qwen 3.5 9B 24/58 done |

---

## Quickstart

All commands are run from `rsna-knee/` with the virtual environment active (`.venv\Scripts\activate`).

```powershell
# 1. Build the 500-study dev subset (hard-links DICOMs, writes folds)
python src/make_subset.py --n 500

# 2. Convert every series to a 32x256x256 uint8 volume
python src/preprocess.py --root E:/Documents/Projects/data/subset

# 3a. Smoke test: 1 fold, 2 epochs, 20 studies
python src/train.py --config configs/base.yaml --set epochs=2 limit_studies=20 folds=[0] out_dir=E:/Documents/Projects/rsna-knee/runs/smoke

# 3b. Full 5-fold cross-validation
python src/train.py --config configs/base.yaml

# 4. Predict the test set with the fold ensemble (the script expands the glob itself, so keep the quotes)
python src/infer.py --data E:/Documents/Projects/data/rsna-knee-abnormality-detection --ckpts "runs/base/fold*.pt" --out submission.csv
```

---

## Repository layout

```
E:\Documents\Projects
├── data\
│   ├── extract-log.txt                     log of the archive extraction
│   ├── rsna-knee-abnormality-detection\    full Kaggle download
│   │   ├── train.csv                       one row per study: StudyInstanceUID, Report, 12 labels
│   │   ├── train_series.csv                one row per series: plane, fat suppression, fluid sensitivity
│   │   ├── train_series\<study>\<series>\*.dcm
│   │   ├── test.csv, test_series.csv, test_series\   (local placeholder, 3 studies)
│   │   └── sample_submission.csv
│   └── subset\                             500-study dev subset (output of make_subset + preprocess)
│       ├── train.csv, train_series.csv     same format as the originals, filtered
│       ├── folds.csv                       StudyInstanceUID, fold, gold, lang, stratum
│       ├── skipped_incomplete.csv          studies rejected because DICOMs were missing or empty
│       ├── train_series\                   hard links to the original DICOMs (no extra disk space)
│       ├── volumes\<SeriesInstanceUID>.npy preprocessed (32, 256, 256) uint8 volumes
│       ├── preprocess_log_train.csv        slice count / error per series
│       └── montage_train.png               visual sanity check of 6 random series
└── rsna-knee\
    ├── configs\base.yaml                   training configuration
    ├── docs\MANIFESTO.md                   what has been done so far, and why
    ├── src\
    │   ├── common.py                       target names, view slots
    │   ├── make_subset.py                  stratified subset + folds
    │   ├── preprocess.py                   DICOM series -> fixed-size volume
    │   ├── labels.py                       gold labels (+ optional weak labels)
    │   ├── weak_labels.py                  report -> 12 labels with a local LLM (llama.cpp)
    │   ├── dataset.py                      study -> 6 slots x 16 slice triplets
    │   ├── model.py                        KneeNet
    │   ├── train.py                        k-fold training, OOF evaluation
    │   └── infer.py                        fold-ensemble inference -> submission.csv
    └── runs\                               one folder per experiment (config, checkpoints, OOF, metrics)
```

---

## The data

### What a knee MRI study looks like

Each study has three levels.

```
Study      one patient's knee MRI exam (one visit)              -> the 12 labels belong here
 └─ Series one scan run: one viewing direction + one contrast  (4–12 per study, median 5)
     └─ Slice  one 2D greyscale image, a thin cut through the knee (11–320 per series, median 30)
                = one .dcm file on disk
```

- **Slice.** An MRI scanner doesn't capture a 3D picture in one go. It records a stack of thin 2D cross-sections,
  each roughly 3–4 mm thick, like the pieces of a sliced loaf of bread. Each piece is one DICOM file.
- **Series.** The whole stack of parallel slices from one run, with the same direction and settings throughout.
  Stacked back together, the slices form a 3D block of the knee.
- **Study.** All the series from one exam. Labels are given **per study** ("this knee has an ACL tear"), never per
  slice or per series. The model therefore has to work out for itself where to look.

### Planes: the direction of the cut

| Plane | What you see | Typically best for |
|---|---|---|
| **Sagittal** | the knee from the side | ACL, menisci, effusion |
| **Coronal** | the knee from the front | MCL, menisci, medial/lateral OA |
| **Axial** | the knee from above (a cross-section, like a tree trunk) | patellofemoral joint (PF OA), Baker's cyst |

### Contrast: with or without fat suppression

- **Without fat suppression (FS=0):** fat is bright. Bone marrow is fatty, so bones look light grey. This gives the
  best view of anatomy.
- **With fat suppression (FS=1):** the scanner cancels the fat signal, so **water and fluid light up bright white**.
  This is how effusion, bone bruises (contusion), fluid-filled tears, cysts and fracture oedema show up.

In the subset, the `Fluid_Sensitive` column always equals `Fat_Suppression`. Every fat-suppressed series is a
fluid-sensitive sequence (for example PD-FS or T2-FS), so the code ignores `Fluid_Sensitive`.

### View slots

3 planes × 2 contrasts = **6 slots** (`common.SLOTS`). Every study is mapped onto these 6 slots: missing ones are
masked out, and when a slot has duplicates, the series with the most slices is kept. Counts in the subset
(500 studies, 2,778 series):

| | FS=0 | FS=1 |
|---|---|---|
| **Sagittal** | 597 | 530 |
| **Coronal** | 457 | 527 |
| **Axial** | 136 (rare) | 531 |

### Labels

| Target | Positives among the 58 gold studies |
|---|---|
| ACL | 24 |
| MCL | 9 |
| Medial Meniscus | 26 |
| Lateral Meniscus | 23 |
| Medial OA | 15 |
| Lateral OA | 11 |
| PF OA | 21 |
| Effusion | 35 |
| Synovitis | 27 |
| Baker's | 12 |
| Contusion | 19 |
| Fracture | 18 |

**Most of the dataset is unlabelled.** Of the 4,407 training studies (24,371 series), only **58 (1.3%) have all 12
labels**. No study has a partial set of labels, so "gold" is all or nothing. Every study, though, comes with a
**free-text radiology report**, in many languages. Subset distribution according to `langdetect`: English 201,
Spanish 77, Turkish 61, Croatian 45, Greek 35, German 29, Bulgarian 25, Dutch 18, French 9. Turning these reports into
weak labels is the obvious way to use the other 98.7% of the data (see [next steps](#known-issues-and-next-steps)).

The local `test.csv` holds only 3 studies (15 series). It is a placeholder, and the real test set is hidden on Kaggle.

---

## Pipeline

### 1. Dev subset: [`make_subset.py`](src/make_subset.py)

The full dataset is too large to iterate on quickly, so a 500-study working set is built.

1. Detects each report's language (`langdetect`, first 400 characters). Languages with fewer than 30 studies are
   merged into `other`.
2. Buckets each study by number of series (`<=4`, `5`, `6`, `>=7`). The stratum is `language|series-bucket`.
3. Keeps **every gold study** whose DICOMs are complete: every series folder exists and has non-empty `.dcm` files.
   This catches files left half-written by the extraction.
4. Fills the remaining slots with unlabelled studies, allocated proportionally across strata (largest-remainder
   method) and drawn at random (seed 42). Incomplete studies are skipped and logged.
5. Assigns 5 folds with `StratifiedKFold`:
   - gold studies are stratified by number of positive findings (capped at 4);
   - unlabelled studies are stratified by language;
   - classes too rare to split are merged into `rare`.
6. **Hard-links** the DICOMs into `data/subset/train_series`, giving a Kaggle-style layout at zero extra disk cost.

### 2. Preprocessing: [`preprocess.py`](src/preprocess.py)

Each series becomes one `(32, 256, 256)` uint8 volume:

1. **Read** every `.dcm`, apply `RescaleSlope` / `RescaleIntercept`, and skip unreadable files.
2. **Sort slices by physical position.** Each slice's `ImagePositionPatient` is projected onto the slice normal
   (the cross product of the two `ImageOrientationPatient` vectors). If the tags are missing, it falls back to
   `InstanceNumber`. File order on disk is not reliable.
3. **Pad each slice to a square** (preserving the in-plane aspect ratio), then resize it to 256×256.
4. **Resample the stack to 32 slices** with trilinear interpolation. A 30-slice series barely changes, a 320-slice
   (3D) series is downsampled 10×, and an 11-slice series is upsampled about 3×.
5. **Normalise intensity.** Clip to the 0.5th–99.5th percentile of non-zero voxels, rescale to [0, 1], and store as
   uint8.

The script runs in parallel (`--workers 8`), skips volumes that already exist (so it is resumable), and writes
`preprocess_log_<split>.csv` (slice count and error per series) plus a `montage_<split>.png` contact sheet.
[`infer.py`](src/infer.py) calls the same `load_volume` on the fly for the test set, so training and test images are
processed identically.

### 3. Labels: [`labels.py`](src/labels.py)

`get_labels` returns a label table (studies × 12 targets, NaN = unknown) and a **per-study loss weight**:

- gold studies: weight **1.0**;
- unlabelled studies: weight **0.0**, so they are currently excluded from training;
- if `weak_labels` is set to a CSV (`StudyInstanceUID` + 12 probability columns, for example produced by an LLM
  reading the reports), its values fill the missing labels and those studies get weight `weak_weight` (default 0.5).

### 4. Dataset: [`dataset.py`](src/dataset.py)

`KneeDataset` turns one study into a model input.

**Series selection.** `pick_series` maps each study to `{slot: SeriesInstanceUID}`. When a slot has several series,
it keeps the one with the most original slices.

**2.5D slice triplets.** From each 32-slice volume:

```
volume (32 slices):  0 1 2 3 4 5 6 7 8 ... 31
                     └─┬─┘ └─┬─┘ └─┬─┘
                      [0,1,2] [2,3,4] [4,5,6] ...   -> 16 evenly spaced triplets
                         ↓
       one triplet = one 3-channel image:
       channel 1 = slice before, channel 2 = centre slice, channel 3 = slice after
```

The image encoder was pretrained on RGB photos and expects 3 channels. MRI is greyscale, so the channels instead
hold *slice before / centre slice / slice after*. This gives a 2D network some depth context: a tear running across
neighbouring slices looks different from noise on a single slice. Because the triplets overlap, the 16 of them
cover almost all 32 slices.

**Augmentation (training only).** These are deliberately mild:

- random 224×224 crop out of 256×256;
- random gamma `exp(U(-0.25, 0.25))`;
- brightness ×`U(0.9, 1.1)` and offset `U(-0.05, 0.05)`.

There are **no flips**: they would swap medial and lateral anatomy, which several targets depend on. Validation
and test use a centre crop.

**Normalisation.** `x / 255`, then `(x - 0.5) / 0.25`.

**Output per study.**

| Key | Shape | Meaning |
|---|---|---|
| `x` | `(6, 16, 3, 224, 224)` | 6 slots × 16 triplets × 3 channels × pixels (zeros for missing slots) |
| `mask` | `(6,)` bool | which slots hold a series |
| `y` | `(12,)` | labels (NaN replaced by 0) |
| `y_mask` | `(12,)` bool | which labels are known |
| `w` | scalar | study loss weight |

---

## The model

[`model.py`](src/model.py) defines **`KneeNet`**: a pretrained image encoder with a small custom head on top.

### Input and output

- **Input:** `x` of shape `(B, 6, 16, 3, 224, 224)` and `mask` of shape `(B, 6)`.
- **Output:** `(B, 12)` logits. A sigmoid gives **independent** probabilities for the 12 findings. It is a
  multi-label problem (a knee can have several findings at once), so there is no softmax across findings.

### Architecture

```
x (B, 6 slots, 16 triplets, 3, 224, 224)
│   drop empty slots, flatten -> N independent 224x224 images
▼
[1] ConvNeXt-Tiny encoder ──────────────► (N, 768)       one vector per triplet
│   reshape + learned slice-position embedding
▼   (B, 6, 16, 768)
[2] Gated attention over the 16 slices ─► (B, 6, 768)    one vector per series
│   + learned slot embedding, LayerNorm
▼
[3] Per-finding attention over 6 series ► (B, 12, 768)   one vector per finding
│   dropout 0.2
▼
[4] 12 per-finding linear classifiers ──► (B, 12)        logits -> sigmoid -> probabilities
```

#### [1] Slice encoder: ConvNeXt-Tiny

- **What it is:** a pure convolutional network from 2022. It is a ResNet modernised with design choices borrowed
  from vision transformers. Loaded through `timm`.
- **Weights:** `convnext_tiny.in12k_ft_in1k`. Pretrained on ImageNet-12k (an 11,821-class subset of ImageNet-22k),
  then fine-tuned on ImageNet-1k. These are natural photos, but low-level features (edges, textures, shapes)
  transfer well to MRI.
- **Structure:**
  - a 4×4, stride-4 stem turns the 224×224 image into a 56×56 grid;
  - 4 stages with **3 / 3 / 9 / 3 blocks** and **96 / 192 / 384 / 768 channels**;
  - between stages, a downsampling layer halves the resolution (56 → 28 → 14 → 7).
- **Each block:** 7×7 depthwise conv → LayerNorm → 1×1 conv to 4× channels → GELU → 1×1 conv back → residual add.
- **Output:** global average pooling of the 7×7 map, then LayerNorm, gives a **768-d vector**. The ImageNet
  classifier is removed (`num_classes=0`).
- **Efficiency:** missing slots are never encoded (`x[mask]`). Cost is about 4.5 GFLOPs per image, and a full
  study is up to 96 images, which is why training uses gradient checkpointing and gradient accumulation.
- **Paper:** [Liu et al., *A ConvNet for the 2020s*, CVPR 2022](https://arxiv.org/abs/2201.03545)

#### [2] Slice pooling: gated attention MIL

This is a multiple-instance learning problem. A series is a "bag" of 16 slice vectors with no per-slice labels, and
a lesion may be visible on only 2 of the 16 slices, so plain averaging would dilute it.

- A learned **slice-position embedding** (16 × 768) is added first, so the model knows whether a slice is at the
  edge or the centre of the knee.
- Each slice vector `h` gets a score `a = wᵀ (tanh(V h) ⊙ sigmoid(U h))`, with `V, U: 768 → 256` and
  `w: 256 → 1`. The sigmoid branch is a learned gate, which lets the score express a non-linear notion of
  relevance.
- A softmax over the 16 slices turns the scores into weights, and the series vector is the weighted sum.
- The same pooling weights are shared across all slots and all findings.
- **Paper:** [Ilse, Tomczak & Welling, *Attention-based Deep Multiple Instance Learning*, ICML 2018](https://arxiv.org/abs/1802.04712)

#### [3] Series pooling: per-finding attention

This is a custom component, not taken from a specific paper.

- A learned **slot embedding** (6 × 768) is added to each series vector, so the model knows which view it is
  looking at (for example sagittal, fat-suppressed). A LayerNorm follows.
- A linear layer `768 → 12` gives every series **one score per finding**.
- Missing slots are filled with −10⁴ before the softmax over the 6 slots, so their weight is effectively zero.
- Each finding gets its own weighted average of the series vectors. The ACL output can learn to rely on the
  sagittal views and the MCL output on the coronal views.

#### [4] Output head

Each finding has its own 768-d weight vector and bias: `logit_t = ⟨dropout(pooled_t), w_t⟩ + b_t`.

### Parameters

Counted from `runs/smoke/fold0.pt`:

| Component | Parameters |
|---|---|
| ConvNeXt-Tiny encoder | 27,820,128 |
| Slice-position embedding | 12,288 |
| Gated attention (slice pooling) | 393,985 |
| Slot embedding + LayerNorm | 6,144 |
| Per-finding attention + output head | 18,456 |
| **Total** | **28,251,001** (~113 MB checkpoint in fp32) |

98.5% of the capacity is pretrained, and the custom head adds only 0.43 M parameters. That balance matters with only
58 labelled studies: a large model trained from scratch would simply memorise them.

---

## Training

[`train.py`](src/train.py) runs k-fold cross-validation. One model is trained per fold.

- **Split:**
  - train = studies outside the validation fold with weight > 0 (currently the gold studies only, about 46 per fold);
  - validation = **gold studies only** in the held-out fold (about 12).
- **Loss:** binary cross-entropy per finding, computed separately for each one. Unknown labels are masked out, and
  each study is multiplied by its weight `w`:
  `loss = Σ BCE · y_mask · w / Σ (y_mask · w)`.
- **Optimiser:** AdamW with weight decay 0.05 and two parameter groups:
  - encoder lr **5e-5**, so pretrained features aren't destroyed early;
  - head lr **2e-4**.
- **Schedule:** linear warm-up for 1 epoch, then cosine decay to 0, stepped once per optimiser step.
- **Batching:** batch size 2 studies (up to 192 images) × 4 accumulation steps = effective batch 8. Gradients are
  clipped to norm 1.0.
- **Memory:**
  - bf16 autocast on the GPU;
  - gradient checkpointing in the encoder (activations are recomputed during the backward pass instead of being
    stored).
- **Model selection:** after every epoch, the validation **macro-AUC** is computed (the mean ROC-AUC over findings
  that have both classes in the fold). The best epoch's checkpoint is saved as `fold{k}.pt`.
- **Outputs in `out_dir`:**
  - `config.yaml`;
  - `fold{k}.pt` (model state + config);
  - `oof.csv`: out-of-fold predictions for every gold study;
  - `metrics.json`: OOF macro-AUC and per-finding AUC.

**About the metric's reliability.** With about 12 validation studies per fold, per-fold AUCs are very noisy. MCL,
for example, has 9 positives in total, about 2 per fold. The OOF macro-AUC over all 58 gold studies is the more
trustworthy number, and even that has wide error bars.

### Smoke-test result

`runs/smoke` (fold 0, 2 epochs, 20 training studies, 5 validation studies): OOF macro-AUC 0.70. This **only shows
the pipeline works**. With 5 validation studies the number carries no information about model quality.

---

## Inference

[`infer.py`](src/infer.py) predicts the test set with the fold ensemble.

1. Reads `test.csv` / `test_series.csv` and maps each study to its slots.
2. Builds each volume on the fly from DICOM with the same `load_volume` used in preprocessing. Volumes are cached
   with an LRU cache of 16.
3. Runs every checkpoint and **averages the sigmoid probabilities**. A study with no usable series gets 0.5 for every
   finding.
4. Writes `submission.csv` (`StudyInstanceUID` + 12 probability columns).

**On Kaggle:** upload `src/` and the `fold*.pt` files as a Kaggle dataset, then run `infer.py` in an offline
notebook. `--data` defaults to `/kaggle/input/rsna-knee-abnormality-detection`. `pretrained=False` is used when
rebuilding the model, so no weight download is needed.

---

## Configuration reference

[`configs/base.yaml`](configs/base.yaml). Any key can be overridden from the command line, for example
`--set epochs=1 folds=[0]`.

| Key | Default | Meaning |
|---|---|---|
| `data_root` | `E:/Documents/Projects/data/subset` | folder with `train.csv`, `folds.csv`, `volumes/` |
| `out_dir` | `E:/Documents/Projects/rsna-knee/runs/base` | where checkpoints and metrics go |
| `weak_labels` | `null` | optional CSV of report-derived label probabilities |
| `weak_weight` | `0.5` | loss weight for weakly labelled studies |
| `backbone` | `convnext_tiny` | any `timm` model name |
| `pretrained` | `true` | load ImageNet weights |
| `grad_ckpt` | `true` | gradient checkpointing in the encoder |
| `n_slices` | `16` | slice triplets sampled per series |
| `crop` | `224` | crop size out of the 256×256 volume |
| `drop` | `0.2` | dropout before the output head |
| `folds` | `[0, 1, 2, 3, 4]` | folds to train |
| `epochs` | `15` | epochs per fold |
| `batch_size` | `2` | studies per batch |
| `accum_steps` | `4` | gradient accumulation steps |
| `lr` / `backbone_lr` | `2e-4` / `5e-5` | head / encoder learning rate |
| `weight_decay` | `0.05` | AdamW weight decay |
| `warmup_epochs` | `1` | linear warm-up length |
| `num_workers` | `0` | DataLoader workers (each Windows worker loads its own torch + ROCm, ~1 GB) |
| `seed` | `42` | random seed |
| `resume` | `true` | skip folds whose `fold{k}.pt` already exists; only recompute their OOF predictions |
| `limit_studies` | `null` | cap on training studies (for smoke tests) |

---

## Environment

- Python 3.13, virtual environment in `rsna-knee/.venv`
- GPU: AMD Radeon RX 9060 XT, using a ROCm build of PyTorch
- Main packages: `torch 2.13.0+rocm10.0.0`, `timm 1.0.30`, `pydicom 3.0.2` (+ `pylibjpeg 2.1.0` for compressed
  DICOM), `numpy 2.5.2`, `pandas 3.0.6`, `scikit-learn 1.9.1`, `PyYAML 6.0.3`, `langdetect 1.0.9`,
  `matplotlib 3.11.2`

There is no `requirements.txt` yet.

On Windows, DataLoader workers use `spawn`, which is why the volume loader is a picklable class (`npy_loader`)
rather than a closure. Each spawned worker also loads its own copy of torch + ROCm (0.7–1.2 GB). On this 16 GB
machine that ran RAM down to 750 MB, so `num_workers` defaults to 0, which costs about 10% speed.

---

## Known issues and next steps

1. **Gold-only training is near chance.** The 5-fold base run reaches an OOF macro-AUC of 0.554. Only Medial OA
   (0.71) shows clear signal, and ACL, MCL, Baker's, Contusion and Fracture sit at 0.48–0.51. (The first attempt
   was killed by low system RAM. It now runs with `num_workers=0`.)
2. **Only the 58 gold studies are used for training.** Unlabelled studies have weight 0, so about 440 of the 500
   subset studies, and about 4,350 of the full dataset, are ignored. Producing weak labels from the multilingual
   reports (for example with an LLM) and passing them via `weak_labels` is the single biggest expected gain. The
   code path already exists.
3. **Test-time series choice is arbitrary.** When a slot has several series, `pick_series` keeps the longest one
   during training. At test time slice counts aren't computed, so any series in the slot may be picked. Counting
   the `.dcm` files per series folder would fix this cheaply.
4. **Slice attention is shared across findings.** Step [2] computes one set of slice weights per series for all 12
   findings, even though different findings sit on different slices. A per-finding slice attention (the step-[3]
   idea applied to slices) is a natural extension.
5. **Physical scale isn't normalised.** Slices are padded and resized to 256×256 without using `PixelSpacing`, and
   stacks are resampled to 32 slices regardless of slice thickness. The knee's size in pixels therefore varies
   between scanners and protocols.

---

## References

- Liu, Mao, Wu, Feichtenhofer, Darrell, Xie. *A ConvNet for the 2020s.* CVPR 2022.
  [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
- Ilse, Tomczak, Welling. *Attention-based Deep Multiple Instance Learning.* ICML 2018.
  [arXiv:1802.04712](https://arxiv.org/abs/1802.04712)
- Chen, Xu, Zhang, Guestrin. *Training Deep Nets with Sublinear Memory Cost* (gradient checkpointing). 2016.
  [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)
- Loshchilov, Hutter. *Decoupled Weight Decay Regularization* (AdamW). ICLR 2019.
  [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Pretrained weights: [timm/convnext_tiny.in12k_ft_in1k](https://huggingface.co/timm/convnext_tiny.in12k_ft_in1k)
