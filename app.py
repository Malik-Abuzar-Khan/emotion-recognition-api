from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # ✅ Allow requests from Android app or any origin

# Load saved models
try:
    model = pickle.load(open("emotion_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model files: {e}")

@app.route('/')
def home():
    return jsonify({"message": "Emotion Recognition API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Transform text into vector
    text_vec = vectorizer.transform([text])

    # Predict emotion
    prediction = model.predict(text_vec)
    emotion = label_encoder.inverse_transform(prediction)[0]

    # Confidence calculation
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_vec)
        confidence = float(np.max(probabilities))
    else:
        confidence = 1.0

    return jsonify({
        "emotion": emotion,
        "confidence": confidence
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # ✅ Render assigns a dynamic port
    app.run(host="0.0.0.0", port=port)