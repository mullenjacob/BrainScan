# BrainScan — Brain Tumor MRI Classifier

A PyTorch image classifier that identifies brain tumor types from MRI scans,
trained on the Kaggle [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
(glioma, meningioma, pituitary tumor, and no tumor), designed to run inference
on an NVIDIA Jetson Orin Nano.

> **Disclaimer:** this is a research/educational project, not a certified
> medical device. Never use it (or any hobby ML project) to make real
> clinical decisions.

![BrainScan architecture](docs/architecture.svg)

## Table of contents

- [Project status (read this first)](#project-status-read-this-first)
- [What's in this project](#whats-in-this-project)
- [How this is meant to be used](#how-this-is-meant-to-be-used)
- [1. Getting the dataset](#1-getting-the-dataset)
- [2. Setting up the Jetson environment](#2-setting-up-the-jetson-environment)
- [3. Training](#3-training)
- [4. Running inference](#4-running-inference)
- [5. Optional: exporting to ONNX / TensorRT](#5-optional-exporting-to-onnx--tensorrt-for-faster-inference)
- [Notes on model choice](#notes-on-model-choice)
- [Improving accuracy, and what's honestly out of reach](#improving-accuracy-and-whats-honestly-out-of-reach)
- [Grad-CAM box precision](#grad-cam-box-precision-and-why-more-data-wasnt-the-fix)
- [Guarding against non-MRI uploads](#guarding-against-non-mri-uploads)
- [Closing the FLAIR/T2 sequence gap](#closing-the-flairt2-sequence-gap-glioma-specifically)
- [Confidence-aware results](#confidence-aware-results)
- [Robustness](#robustness)
- [Pituitary detection and image resolution](#pituitary-detection-and-image-resolution-confirmed-improvement)

## Project status (read this first)

- **In-distribution accuracy (T1-weighted contrast-enhanced MRI, matching
  the primary training data): ~96%** on held-out test data.
- **External validation on a genuinely independent, held-out FLAIR dataset
  (glioma vs. notumor only): ~90%** — see `src/evaluate_external.py`. This
  number matters more than it might look: it's measuring generalization to
  an MRI sequence the model wasn't originally trained on, using patients
  never seen during training, not just re-confirming performance on data
  similar to what it already trained on.
- **Known gap:** the FLAIR fix above only covers glioma/notumor. Meningioma
  and pituitary tumors on FLAIR-sequence images are untested and may not
  generalize as well — no dataset was found/added to specifically validate
  that.
- **Low-confidence uploads are flagged explicitly in the UI** rather than
  displayed with the same visual confidence as a clear result — see
  "Confidence-aware results" below. The goal isn't "never wrong" (no
  classifier trained on ~10k images achieves that) — it's "never
  confidently wrong-looking."
- **Robustness-tested** against oversized images, alpha-channel PNGs,
  grayscale-mode images, corrupted files, and missing uploads — none of
  these crash the app; see "Robustness" below.

## What's in this project

```
brain_tumor_classifier/
├── config.yaml              # all hyperparameters / paths in one place
├── requirements.txt          # everything except torch/torchvision
├── docs/
│   └── architecture.svg      # pipeline diagram (see below)
├── data/                     # datasets go here (empty until you download them)
├── models/                   # trained checkpoints + plots land here
├── scripts/
│   ├── download_dataset.py       # pulls the primary Kaggle dataset down
│   ├── download_lgg_dataset.py   # pulls the independent FLAIR/LGG dataset down
│   ├── prepare_lgg_data.py       # labels + splits the LGG data by patient
│   └── setup_jetson.sh           # sets up CUDA PyTorch on the Jetson
├── src/
│   ├── dataset.py            # data loading + augmentation
│   ├── model.py               # model architecture (MobileNetV3 / ResNet18 / EfficientNet-B0)
│   ├── train.py               # training loop, saves best checkpoint + plots
│   ├── inference.py           # single-image CLI prediction (with TTA)
│   ├── gradcam.py              # Grad-CAM heatmap + bounding box extraction
│   ├── input_guard.py           # heuristic check: "does this look like an MRI?"
│   ├── evaluate_external.py     # honest FLAIR-generalization check (held-out LGG patients)
│   └── export_onnx.py         # optional ONNX export for faster inference
└── app/
    ├── server.py               # Flask web app: upload an MRI, get a prediction + box overlay
    ├── templates/index.html
    └── static/style.css        # dark/neon UI
```

## How this is meant to be used

Given your setup (laptop → VS Code Remote-SSH → Jetson Orin Nano, JetPack 6,
Python 3.10), the intended flow is:

1. Do everything **on the Jetson**, through your Remote-SSH terminal in VS Code.
2. Download the dataset, train the model, then run the web app — all on-device.
3. Training a MobileNetV3-based model on the combined dataset (~10,000+
   images once the FLAIR/LGG data is added — see "Closing the FLAIR/T2
   sequence gap" below) is well within the Orin Nano's GPU budget (expect
   roughly 1-2 minutes per epoch).

If you'd rather train on a beefier machine (laptop/desktop GPU or Colab) and
just deploy the resulting checkpoint to the Jetson, that works fine too — just
copy `models/best_model.pth` over afterward and skip straight to running the app.

**This project isn't Jetson-locked.** Every part of it -- training, inference,
the web app -- is standard PyTorch/Flask with no Jetson-specific code paths;
it runs the same way on a laptop or desktop, with or without a GPU. The
Jetson Orin Nano is the intended edge-deployment target (and the reason
lightweight backbones like MobileNetV3 were chosen -- see "Notes on model
choice" below), and `scripts/setup_jetson.sh` exists specifically to install
a CUDA-enabled PyTorch build matched to the Jetson's unusual ARM64
architecture. On a laptop or desktop, skip that script entirely and just
`pip install torch torchvision` normally (add a CUDA index URL if you have
an NVIDIA GPU, or leave it as the default CPU build otherwise), then follow
the rest of this README as written.

---

## 1. Getting the dataset

Kaggle requires authentication to download datasets via the API, so this step
needs a one-time setup on your end:

1. Go to https://www.kaggle.com/settings and click **Create New Token** under
   the API section. This downloads a `kaggle.json` file.
2. Move it to `~/.kaggle/kaggle.json` on whichever machine you're downloading
   from (the Jetson, if you're doing everything on-device), and lock down its
   permissions:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Install the `kaggle` CLI (it's in `requirements.txt`) and run:
   ```bash
   python scripts/download_dataset.py
   ```
   This downloads and unzips the dataset into `data/`. You should end up with:
   ```
   data/Training/glioma/...
   data/Training/meningioma/...
   data/Training/notumor/...
   data/Training/pituitary/...
   data/Testing/glioma/...
   data/Testing/meningioma/...
   data/Testing/notumor/...
   data/Testing/pituitary/...
   ```
   (Exact class folder names may vary slightly by dataset version — the code
   auto-detects whatever folder names are present, so it doesn't matter if
   they're not exactly these.)

   If the zip extracts with a different top-level structure (e.g. an extra
   nested folder), either move things around to match the above, or just edit
   `data.train_dir` / `data.test_dir` in `config.yaml` to point at wherever
   things actually landed.

**Alternative (manual) download:** if you'd rather not set up the API, just
download the dataset zip directly from the Kaggle page in a browser and unzip
it into `data/` yourself, matching the structure above.

---

## 2. Setting up the Jetson environment

**Important:** plain `pip install torch` on the Jetson's ARM64 architecture
silently installs a CPU-only build — it will run, but won't use the GPU.
You need Jetson-specific wheels.

From your VS Code Remote-SSH terminal, in the project folder:

```bash
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh
```

This script:
- Creates a `.venv` virtual environment
- Installs a CUDA-enabled PyTorch + torchvision build matched to JetPack 6.x /
  Python 3.10 (via the Jetson AI Lab pip index)
- Installs the rest of `requirements.txt`
- Prints a check confirming `torch.cuda.is_available()` is `True`

If the primary install method fails (this can happen if your JetPack point
version doesn't line up exactly, e.g. 6.0 vs 6.1 vs 6.2), the script prints
fallback instructions pointing you to NVIDIA's direct wheel repository —
you'll need to grab the wheel matching your specific JetPack version from
there. The exact wheel URLs shift with each JetPack release, so if in doubt,
search "PyTorch for Jetson" on the NVIDIA Developer Forums for the current link.

After setup, activate the environment in future sessions with:
```bash
source .venv/bin/activate
```

---

## 3. Training

With the dataset in `data/` and the environment set up:

```bash
python src/train.py --config config.yaml
```

This will:
- Split `data/Training` into train/validation sets (85/15 by default)
- Fine-tune a MobileNetV3-Small (ImageNet-pretrained) for up to 20 epochs,
  with early stopping if validation accuracy stalls for 5 epochs
- Evaluate the best checkpoint on the untouched `data/Testing` set
- Save to `models/`:
  - `best_model.pth` — the checkpoint (weights + class names + config)
  - `training_history.png` — loss/accuracy curves
  - `confusion_matrix.png` — per-class performance on the test set

Tweak `config.yaml` to change the backbone (`mobilenet_v3_small`,
`mobilenet_v3_large`, or `resnet18` for higher accuracy at more compute cost),
batch size, learning rate, epochs, or image size.

---

## 4. Running inference

**Single image, from the command line:**
```bash
python src/inference.py --image path/to/scan.jpg --checkpoint models/best_model.pth
```
Prints a ranked confidence breakdown across all classes.

**Web app (upload + classify, with a Grad-CAM box overlay):**
```bash
python app/server.py --checkpoint models/best_model.pth --port 5000
```
Then open `http://<jetson-ip>:5000` in a browser (or forward port 5000 through
your VS Code Remote-SSH connection and open `http://localhost:5000` on your
laptop). Drag and drop an MRI image to get a prediction with a per-class
confidence breakdown — and, when a tumor class is predicted, a bounding box
around the region(s) the model focused on (via Grad-CAM). No box is drawn for
"no tumor" predictions.

**Important:** Grad-CAM shows where the model's attention concentrated, which
is an interpretability aid, not a validated tumor boundary or measurement —
treat it as "here's roughly what drove this prediction," not a segmentation.

If you upload something that doesn't look like a grayscale, skull-cropped MRI
scan (a color photo, a screenshot, etc.), the app catches this with a
heuristic check and shows a message plus a real example image from the
training set instead of running the classifier on it.

---

## 5. Optional: exporting to ONNX / TensorRT for faster inference

```bash
python src/export_onnx.py --checkpoint models/best_model.pth --out models/brain_tumor_model.onnx
```

For maximum inference speed on the Jetson, you can then build a TensorRT
engine from the ONNX file using the `trtexec` tool that ships with JetPack:

```bash
trtexec --onnx=models/brain_tumor_model.onnx \
        --saveEngine=models/brain_tumor_model.engine \
        --fp16
```

This is optional — MobileNetV3 is already small/fast enough that plain
PyTorch inference is fine for the upload-and-classify use case. TensorRT is
worth it if you want to push toward higher throughput or plan to run several
models concurrently.

---

## Notes on model choice

- **MobileNetV3-Small** (default): fastest, lowest memory, good fit for edge
  deployment on the Orin Nano. Expect solid accuracy (dataset separates the
  four classes fairly well) with fast training.
- **MobileNetV3-Large**: a step up in capacity/accuracy, still edge-friendly.
- **ResNet18**: higher accuracy ceiling, heavier — worth trying if you have
  training time to spare and want to compare, since it's just a config change.
- **EfficientNet-B0**: tends to be the most accurate of the four on this
  dataset, at a modest cost in size/inference speed vs MobileNetV3-Small —
  a good choice if you want to squeeze out more accuracy and don't mind the
  larger model.

Swap backbones by changing `model.backbone` in `config.yaml` and re-running
`train.py` — no other code changes needed.

---

## Improving accuracy, and what's honestly out of reach

A few concrete things in this project push accuracy up without adding new
data:

- **Stratified train/val split** (`src/dataset.py`) — the split now
  guarantees each class is represented in the same proportion in both train
  and val, rather than a plain random split that can (with a 4-class, few-
  thousand-image dataset) leave val slightly skewed toward or away from a
  given class by chance. This makes val_acc a more reliable signal for early
  stopping and model comparison.
- **Slightly stronger, still MRI-appropriate augmentation** — mild random-
  resized-crop, rotation, color jitter, and random erasing. Kept moderate on
  purpose: MRI scans are consistently framed/oriented, so aggressive crops or
  rotations create unrealistic training examples rather than useful variety.
- **Label smoothing** (`train.label_smoothing` in `config.yaml`) — reduces
  overconfidence in the model's predictions, which tends to help
  generalization slightly.
- **EfficientNet-B0 backbone option** — generally the most accurate of the
  four backbones on this dataset (see above).
- **Test-time augmentation (TTA)** at inference (`src/inference.py` and
  `app/server.py`) — every prediction now averages the model's output over
  the image and its horizontal flip. This is free (no retraining) and
  typically worth a small, consistent accuracy bump.

**On adding more datasets:** worth knowing before you go looking — the
Kaggle dataset this project already uses is itself a merge of three older
public datasets (figshare, SARTAJ, and Br35H). Most other "brain tumor MRI"
datasets you'll find on Kaggle are repackaged versions of those same three
sources. Blindly adding another one risks adding near-duplicate images that
inflate your test accuracy without the model actually generalizing better,
or worse, leaking test-set images into training. If you want genuinely
independent data, look at multi-institutional research archives like
[BraTS](http://braintumorsegmentation.org/) or [TCIA](https://www.cancerimagingarchive.net/)
(the Cancer Imaging Archive) — but note these are typically DICOM,
multi-sequence (T1/T2/FLAIR/contrast), and set up for tumor *segmentation*
rather than whole-image classification, so incorporating them is a
substantially bigger project than swapping a training folder.

**On "recognizing early conditions":** this is worth being direct about.
This dataset (like essentially all public brain tumor MRI classification
datasets) only has labels for already-diagnosed, clearly visible tumors —
there's no "early/subtle/small" annotation to train against, because nobody
labeled the data that way. Detecting an early-stage tumor is, in real
radiology, one of the hardest and highest-stakes tasks there is — it often
requires multiple MRI sequences, contrast agents, and sometimes comparison
against follow-up scans over time, and remains genuinely difficult even for
trained specialists. A classification model trained on a single-sequence,
few-thousand-image public dataset isn't a credible way to do that, and
claiming otherwise would be misleading. If early-stage detection is a goal
worth pursuing further, it would mean a different project: segmentation
models on expert-annotated, multi-sequence data (BraTS-style) with proper
clinical validation — a research-grade undertaking, not a config change.

---

## Grad-CAM box precision, and why "more data" wasn't the fix

If you've noticed the Grad-CAM box landing somewhere that doesn't match the
visibly obvious lesion in the image, that's a real limitation worth
understanding rather than just tolerating.

**What was actually wrong:** Grad-CAM's precision depends entirely on which
convolutional layer it's hooked into. The first version of this project
hooked the model's very last conv layer — which, for every backbone here
(MobileNetV3, ResNet18, EfficientNet-B0), has been downsampled all the way to
a 7x7 grid for a 224x224 input. Each cell in that grid represents a 32x32
pixel block of the original image, so boxes come out coarse, blocky, and
sometimes miss the visually obvious feature entirely — the localization
resolution is just too low, independent of how much data the model saw.

**The fix:** `src/gradcam.py`'s `find_target_conv_layer()` now probes the
model's actual conv layers with a dummy forward pass and picks the last one
that still has at least a 14x14 spatial resolution, instead of blindly using
the literal last layer. This doubles the effective resolution (16x16 pixel
blocks instead of 32x32) across all four backbones, while staying deep
enough in the network to still reflect the class decision. This required no
retraining — it's purely a change to how the existing trained model is
visualized after the fact.

**Being honest about what this does and doesn't fix:** better resolution
means tighter, less blocky boxes — it does not guarantee the model is
"looking at" the lesion a human would circle. Classifiers trained on modest
datasets can and do learn shortcuts (asymmetry between hemispheres, skull
shape, scan artifacts) that happen to correlate with the right class label
without the model's attention landing on the lesion itself. If boxes still
look off after this fix, that's a genuine signal worth taking seriously about
the model's actual decision process — not something to paper over by adding
a "confidence" caveat and moving on.

---

## Guarding against non-MRI uploads

`src/input_guard.py`'s `looks_like_mri()` runs before the classifier on every
upload and checks two simple pixel statistics: whether the image is
essentially grayscale (MRI scans have near-zero color saturation even when
saved as RGB/JPEG, unlike ordinary color photos), and whether the image
corners are dark (MRI scans in this dataset are cropped against a black
background). If either check fails, the app shows a message and a real
example image from the training set instead of running the classifier.

This is a heuristic, not a trained detector — it's a couple of pixel
statistics, not a model that learned what "not an MRI" looks like. It will
reliably catch obviously-wrong uploads (photos, screenshots) but could in
principle be fooled by other grayscale images, or rarely flag an unusual
but legitimate scan. It requires no additional dataset or training.

---

## Closing the FLAIR/T2 sequence gap (glioma specifically)

Extensive real-world testing surfaced a clear pattern: the model does
reasonably well on meningioma and pituitary cases, but consistently fails on
what look like glioma cases — often calling an obviously abnormal image
"notumor." The likely reason: **this project's original training data is
entirely T1-weighted, contrast-enhanced MRI** (that's what the figshare
source behind glioma/meningioma/pituitary actually is). Gliomas are diffuse
and infiltrative rather than sharply-bordered, so radiologists specifically
favor **FLAIR** sequences to visualize them — meaning real-world "glioma"
test images are disproportionately likely to be FLAIR, a sequence type the
model has literally never seen a single training example of. Meningioma and
pituitary tumors, being more discrete and well-circumscribed, show up
clearly on T1 contrast-enhanced imaging — matching what the model actually
trained on — which is plausibly why those classes perform better.

To close this gap, three new scripts pull in a **second, genuinely
independent dataset**: the Kaggle **LGG Segmentation Dataset**
(`mateuszbuda/lgg-mri-segmentation`), sourced from TCGA's lower-grade-glioma
collection. This matters for two reasons: it's FLAIR-based (the exact
sequence type missing from training), and it's a completely different
source from figshare/SARTAJ/Br35H — no overlap risk, unlike grabbing another
generic Kaggle "brain tumor" dataset.

**Scope, honestly:** this dataset only contains glioma-patient scans (with
some tumor-free slices per patient), so it can only help the glioma/notumor
distinction — it has no meningioma or pituitary examples. It won't close a
FLAIR gap for those two classes; that would need a different data source.

### How to use it

```bash
python scripts/download_lgg_dataset.py    # needs the same Kaggle credentials as before
python scripts/prepare_lgg_data.py
python src/train.py --config config.yaml   # retrain, now including FLAIR glioma/notumor examples
python src/evaluate_external.py --checkpoint models/best_model.pth
```

`prepare_lgg_data.py` uses each slice's segmentation mask to auto-label it
(tumor pixels present -> glioma, mask empty -> notumor), then splits by
**patient**, not by slice, into two groups:

- Most patients' slices get added directly into `data/Training/glioma` and
  `data/Training/notumor`, alongside the existing data.
- A held-out ~20% of patients go into `data/ExternalValidation/` instead —
  kept separate from training on purpose.

`evaluate_external.py` then reports accuracy specifically on that held-out
FLAIR set. This is the honest way to check whether this actually helped:
training accuracy or the existing `Testing` folder's accuracy wouldn't tell
you anything new here, since both are already the T1C+ distribution the
model was already fine on. The external eval script is measuring the thing
that was actually broken.

---

## Confidence-aware results

Every prediction is classified as `high`, `moderate`, or `low` confidence,
based on the top prediction's probability and its margin over the runner-up
class (see `run_prediction()` in `app/server.py`). Low-confidence results get
a visible warning banner in the UI instead of being displayed identically to
a clear-cut result.

This isn't a calibration study — the thresholds are a deliberately
conservative judgment call, not derived from a validation curve. The point
isn't to make the model more accurate; it's to make sure that when the model
genuinely doesn't know, the interface doesn't paper over that with a
confident-looking percentage. A wrong answer that's honestly flagged as
uncertain is a far better failure mode than a wrong answer that looks
authoritative.

## Robustness

The upload path is tested against: images larger than 3000x3000 (auto-capped
to 1024px on the longest side before processing, so a huge upload can't
slow down or hang a request), RGBA/alpha-channel PNGs, grayscale-mode
images, corrupted/non-image files, missing uploads, and degenerate tiny
images — all handled without crashing (either processed normally or
returned as a clean error response). See the robustness test battery used
to verify this for the specific cases covered.

---

## Pituitary detection and image resolution (confirmed improvement)

Real-world testing repeatedly showed sagittal images with something pointing
at or highlighting the pituitary/sella region getting classified as
"notumor." Worth noting: some of these test images were annotated teaching
images (arrows, letters), and that style is sometimes used to point out
*normal* anatomy rather than a lesion — so not every one of these was
necessarily a confirmed error. But the pattern recurred enough to be worth
taking seriously.

Unlike the FLAIR/glioma gap, this didn't look like a data-quantity problem:
pituitary actually has the *most* training slices of the three tumor types
(1757, vs. 1621 glioma and 1645 meningioma). The likelier explanation was
resolution: the pituitary gland is anatomically small relative to the whole
head, and a pituitary adenoma (especially a "microadenoma") can occupy only
a small region even in the original image. Gliomas and meningiomas are often
large enough to survive being resized down to 224x224 with their key
features intact; a small pituitary lesion may not.

**`config.yaml`'s `img_size` was bumped from 224 to 320** to test this
directly — giving small structures more surviving detail after resizing.
**Result: this held up.** After retraining at 320x320:

- A previously-problematic pituitary case that had scored 95.8% confidence
  with the Grad-CAM box landing on an image watermark instead of the actual
  lesion improved to 98.6% confidence, with the box now landing directly on
  the lesion.
- The held-out FLAIR external validation accuracy (see
  `evaluate_external.py` above) improved from 87.1% to 90.3% — so the
  resolution increase helped generalization broadly, not just the pituitary
  class it was originally targeting, with no measurable cost to
  in-distribution test accuracy (95.9% vs. 96.1% before — within normal
  run-to-run noise).

This is a case where a specific, testable hypothesis about *why* a class was
underperforming (small structures losing detail at low resolution) was
proposed, tested, and confirmed with before/after numbers — not just a
tuning change made and hoped for the best.
