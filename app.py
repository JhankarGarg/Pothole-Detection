from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load trained model
model = YOLO("best.pt")

total_images = 0
total_potholes = 0


@app.route("/")
def home():
    return render_template("index.html")


# IMAGE DETECTION
@app.route("/predict_image", methods=["POST"])
def predict_image():

    import time
    start_time = time.time()

    file = request.files["image"]

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    results = model(img, conf=0.3)

    pothole_count = len(results[0].boxes)

    annotated = results[0].plot()

    _, buffer = cv2.imencode(".jpg", annotated)

    img_base64 = base64.b64encode(buffer).decode("utf-8")

    processing_time = round(time.time() - start_time, 2)

    return jsonify({
        "image": img_base64,
        "pothole_count": pothole_count,
        "processing_time": processing_time
    })


# WEBCAM DETECTION
@app.route("/predict_webcam", methods=["POST"])
def predict_webcam():

    data = request.json["image"]

    img_data = base64.b64decode(data.split(",")[1])

    npimg = np.frombuffer(img_data, dtype=np.uint8)

    frame = cv2.imdecode(npimg, 1)

    results = model(frame, conf=0.25)

    pothole_count = len(results[0].boxes)

    annotated = results[0].plot()

    _, buffer = cv2.imencode(".jpg", annotated)

    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "image": img_base64,
        "pothole_count": pothole_count
    })


# STATS
@app.route("/stats")
def stats():
    return jsonify({
        "total_images": total_images,
        "total_potholes": total_potholes
    })


if __name__ == "__main__":
    app.run(debug=True)