"""
server.py
---------
Flask web app for the brain tumor classifier project.

Upload an MRI image and get a prediction from the trained classifier, with
per-class confidence scores and a bounding box overlay (via Grad-CAM) around
the region(s) that drove the prediction. No box is drawn for "no tumor"
predictions.

Note: Grad-CAM shows where the model's attention concentrated, which is an
interpretability aid, not a validated tumor boundary or measurement.

Usage:
    python app/server.py --checkpoint models/best_model.pth --port 5000

Then open http://<jetson-ip>:5000 in a browser (or http://localhost:5000
if you've port-forwarded 5000 through VS Code Remote-SSH).
"""

import argparse
import base64
import glob
import io
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from model import load_checkpoint_for_inference
from dataset import IMAGENET_MEAN, IMAGENET_STD, clahe_normalize, auto_crop_to_brain, find_brain_crop_box
from gradcam import GradCAM, heatmap_to_boxes, scale_boxes, is_no_tumor_class
from input_guard import looks_like_mri

DATA_TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Training")

app = Flask(__name__)

# ---- Global model state, loaded once at startup ----
STATE = {
    "model": None,
    "class_names": None,
    "img_size": None,
    "device": None,
}
GRADCAM = None  # set in load_model(); wraps STATE["model"] with forward/backward hooks

BOX_COLOR_BGR = (195, 255, 0)  # neon teal, matches the site theme


def load_model(checkpoint_path: str):
    global GRADCAM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, img_size = load_checkpoint_for_inference(checkpoint_path, device)
    STATE.update(model=model, class_names=class_names, img_size=img_size, device=device)
    GRADCAM = GradCAM(model, input_size=img_size)
    print(f"Loaded model from {checkpoint_path} on {device}. Classes: {class_names}")


def preprocess_pil(image: Image.Image):
    # auto_crop_to_brain and CLAHE MUST match what training used (see
    # dataset.py) — applying different preprocessing here than at training
    # time would make predictions worse, not better.
    tf = transforms.Compose([
        transforms.Lambda(auto_crop_to_brain),
        transforms.Lambda(clahe_normalize),
        transforms.Resize((STATE["img_size"], STATE["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(image.convert("RGB")).unsqueeze(0)


MAX_UPLOAD_DIMENSION = 1024  # safety cap, see run_prediction()


def run_prediction(image: Image.Image):
    """
    Runs the classifier, plus Grad-CAM box extraction.

    Uses test-time augmentation (TTA): averages softmax probabilities over the
    original image and its horizontal flip. This is a well-established, free
    accuracy boost (no retraining needed) since it's equivalent to asking the
    model twice from two valid, equally-legitimate views and trusting the
    consensus more than either single view. Grad-CAM is computed on the
    original (unflipped) image only, so box coordinates stay simple to map
    back to the original image.
    """
    image = image.convert("RGB")

    # Safety cap: a huge upload (a multi-thousand-pixel photo, say) would make
    # the crop/CLAHE steps (which run at full resolution before any resizing)
    # slow and memory-heavy — exactly the kind of thing that could hang or
    # visibly stall a live demo. 1024px is still well beyond what the model
    # actually uses internally (it resizes down to the trained input size
    # anyway), so this costs no real accuracy.
    if max(image.size) > MAX_UPLOAD_DIMENSION:
        image.thumbnail((MAX_UPLOAD_DIMENSION, MAX_UPLOAD_DIMENSION))

    tensor = preprocess_pil(image).to(STATE["device"])
    flipped_tensor = torch.flip(tensor, dims=[3])  # horizontal flip on the width axis

    class_names = STATE["class_names"]

    # Grad-CAM needs gradients, so this runs outside torch.no_grad().
    logits = STATE["model"](tensor)
    probs = F.softmax(logits, dim=1).squeeze(0)

    with torch.no_grad():
        flipped_logits = STATE["model"](flipped_tensor)
        flipped_probs = F.softmax(flipped_logits, dim=1).squeeze(0)

    avg_probs = (probs.detach() + flipped_probs) / 2.0
    pred_idx = int(avg_probs.argmax().item())
    pred_label = class_names[pred_idx]

    ranked = sorted(
        zip(class_names, avg_probs.cpu().tolist()), key=lambda x: x[1], reverse=True
    )

    boxes_original = []
    annotated_b64 = None

    if not is_no_tumor_class(pred_label):
        heatmap = GRADCAM.generate(tensor, pred_idx)
        boxes_model_space = heatmap_to_boxes(heatmap, threshold=0.55, min_area_frac=0.003, max_boxes=3)

        # preprocess_pil() crops the image (auto_crop_to_brain) before resizing
        # to model input size, so heatmap-space boxes are relative to the
        # CROPPED region, not the original image. Map them back through both
        # steps: model space -> cropped-image space -> original-image space
        # (adding back the crop's offset), so the overlay lands correctly on
        # the original image we show the person.
        crop_box = find_brain_crop_box(image)
        if crop_box is not None:
            x0, y0, x1, y1 = crop_box
            cropped_size = (x1 - x0, y1 - y0)
        else:
            x0, y0 = 0, 0
            cropped_size = image.size

        boxes_in_cropped_space = scale_boxes(boxes_model_space, (STATE["img_size"], STATE["img_size"]), cropped_size)
        boxes_original = [(x + x0, y + y0, w, h) for (x, y, w, h) in boxes_in_cropped_space]

        if boxes_original:
            annotated = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            for (x, y, w, h) in boxes_original:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), BOX_COLOR_BGR, 3)
            ok, buf = cv2.imencode(".jpg", annotated)
            if ok:
                annotated_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    top1_prob = ranked[0][1]
    top2_prob = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top1_prob - top2_prob

    # These thresholds aren't from a formal calibration study — they're a
    # deliberately conservative, honest-by-default line: better to flag a
    # correct-but-modest-confidence prediction as "uncertain" than to let a
    # genuinely ambiguous one look confidently authoritative.
    if top1_prob >= 0.85 and margin >= 0.30:
        confidence_level = "high"
    elif top1_prob >= 0.60 and margin >= 0.15:
        confidence_level = "moderate"
    else:
        confidence_level = "low"

    return {
        "label": pred_label,
        "confidence": top1_prob,
        "confidence_level": confidence_level,
        "all_probs": {cls: p for cls, p in ranked},
        "boxes": boxes_original,           # [[x, y, w, h], ...] in original image pixel coords
        "annotated_image": annotated_b64,  # base64 JPEG with boxes drawn, or null if none/no-tumor
    }


@app.route("/")
def index():
    return render_template("index.html", class_names=STATE["class_names"] or [])


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        image = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    is_mri, reason = looks_like_mri(image)
    if not is_mri:
        return jsonify({
            "not_mri": True,
            "message": f"{reason} Please upload a brain MRI scan instead.",
        })

    result = run_prediction(image)
    return jsonify(result)


@app.route("/reference_image")
def reference_image():
    """Serves a random real sample image from the training set, so the UI
    can show the person roughly what a valid upload looks like. Randomized
    (rather than always the same file) since the training data includes
    scans from multiple angles (from above, from the side, from the front)
    — always showing the same one image risked implying only that specific
    angle/style is acceptable, which isn't true."""
    candidates = glob.glob(os.path.join(DATA_TRAINING_DIR, "*", "*.jpg"))
    if not candidates:
        return jsonify({"error": "No reference image available — dataset not downloaded."}), 404
    return send_file(random.choice(candidates), mimetype="image/jpeg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best_model.pth")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    load_model(args.checkpoint)
    app.run(host=args.host, port=args.port, threaded=True)
