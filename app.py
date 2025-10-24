from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load saved models
model = pickle.load(open("emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

app = Flask(__name__)

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

    # Get prediction probabilities if available
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_vec)
        confidence = float(np.max(probabilities))  # Highest probability
    else:
        confidence = 1.0  # fallback if model doesn’t support probabilities

    return jsonify({
        "emotion": emotion,
        "confidence": confidence
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)