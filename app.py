from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import os

app = Flask(__name__)

# Load the TFLite model
MODEL_PATH = "face_emotionModel_compat.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors info
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define emotion labels (adjust these if your model uses different ones)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def predict_emotion(image):
    """
    Takes an image (as np.array), preprocesses it, and returns the predicted emotion.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    img = np.expand_dims(normalized, axis=(0, -1)).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = int(np.argmax(output_data))
    emotion = EMOTIONS[predicted_class]
    return emotion

@app.route("/")
def index():
    return render_template("index.html")  # your front page

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No select

