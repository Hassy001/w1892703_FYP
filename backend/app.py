from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import base64
import cv2

from preprocess import load_image
from model import load_model
from gradcam import GradCAM, overlay_cam

app = Flask(__name__)
CORS(app)

LABEL_MAP = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma (pre-cancerous)",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis (seborrheic keratosis)",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus (benign mole)",
    "vasc": "Vascular lesion (angioma)"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, classes = load_model(device)

target_layer = model.layer4
gradcam = GradCAM(model, target_layer)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(device)
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image_file = request.files["image"]
        image, tensor = load_image(image_file)
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_idx = int(torch.argmax(probs).item())
            conf = float(probs[pred_idx].item())

        short_label = classes[pred_idx]
        label = LABEL_MAP.get(short_label, short_label)
        uncertain = False

        cam = gradcam.generate(tensor, pred_idx)
        overlay = overlay_cam(image, cam)

        ok, buffer = cv2.imencode(".png", overlay)
        if not ok:
            return jsonify({"error": "Failed to encode overlay image"}), 500

        overlay_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "label": label,
            "code": short_label,
            "confidence": round(conf, 4),
            "uncertain": bool(uncertain),
            "gradcam_overlay": overlay_b64
        })

    except Exception as e:
        print("PREDICT ERROR:", repr(e))
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
