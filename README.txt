Emotion Recognition Flask API - README

---------------------------------------------------------------
Overview
---------------------------------------------------------------
This is the backend service for the Android app "Emotion Recognition from Text".
It uses a trained NLP model to predict emotions such as joy, sadness, anger, and others
based on the text input received from the Android app.

---------------------------------------------------------------
Features
---------------------------------------------------------------
- REST API endpoint `/predict` for emotion detection
- Lightweight model trained on the GoEmotions dataset
- Uses scikit-learn / TensorFlow Lite model
- JSON input and output for easy Android integration

---------------------------------------------------------------
Setup Instructions
---------------------------------------------------------------

1. Install Dependencies
   Make sure Python 3.8+ is installed.
   Then, open your terminal in this folder and run:
       pip install -r requirements.txt

2. Run the Flask Server
   Start the backend server with:
       python app.py

   By default, the server runs at:
       http://127.0.0.1:5000/

---------------------------------------------------------------
API Endpoint Details
---------------------------------------------------------------

POST /predict

Example Request (JSON):
{
  "text": "I am feeling great today!"
}

Example Response (JSON):
{
  "emotion": "joy",
  "confidence": 0.93
}

---------------------------------------------------------------
Model Details
---------------------------------------------------------------
- Dataset: GoEmotions (Google - 58k Reddit comments, 27 emotion labels)
- Algorithm: Multinomial Naive Bayes using TF-IDF features
- Framework: scikit-learn
- Serialized model format: Pickle (.pkl)

---------------------------------------------------------------
Testing
---------------------------------------------------------------
To test manually using curl or Postman:

Command:
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"I am so happy today!\"}"

Expected Output:
{"emotion": "joy", "confidence": 0.94}

---------------------------------------------------------------
Integration with Android App
---------------------------------------------------------------
In your Android project (TryItActivity.kt), update the API URL if required:

val apiUrl = "http://10.0.2.2:5000/predict"

Ensure Flask server is running before testing from Android app.

Supervisor
---------------------------------------------------------------
**Name*: Sir Mr.Muhammad Imran Afzal

---------------------------------------------------------------
Developer
---------------------------------------------------------------
Author: Abuzar Khan
Final Year Project - Emotion Recognition from Text in Android