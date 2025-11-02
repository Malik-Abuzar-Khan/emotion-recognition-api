from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import json
import firebase_admin
from firebase_admin import credentials, auth, firestore

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)

# ✅ Initialize Firebase Admin SDK
try:
    # If using Render Environment Variable:
    if "FIREBASE_CREDENTIALS" in os.environ:
        cred_dict = json.loads(os.environ.get("FIREBASE_CREDENTIALS"))
        cred = credentials.Certificate(cred_dict)
    else:
        # Local testing (JSON file in same folder)
        cred = credentials.Certificate("firebase_admin_key.json")

    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase Admin initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase Admin: {e}")

# ✅ Load ML Models
try:
    model = pickle.load(open("emotion_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"Error loading model files: {e}")

# ------------------------------------------------------------
# 🔹 ROUTES
# ------------------------------------------------------------
@app.route('/')
def home():
    return jsonify({"message": "Emotion Recognition API is running!"})

# 🔹 Emotion Prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    emotion = label_encoder.inverse_transform(prediction)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_vec)
        confidence = float(np.max(probabilities))
    else:
        confidence = 1.0

    return jsonify({
        "emotion": emotion,
        "confidence": confidence
    })

# ------------------------------------------------------------
# 🔹 ADMIN FEATURES
# ------------------------------------------------------------

# ✅ Get all users
@app.route('/admin/get_users', methods=['GET'])
def get_users():
    try:
        users = []
        docs = db.collection("users").get()
        for doc in docs:
            user = doc.to_dict()
            user["uid"] = doc.id
            users.append(user)
        return jsonify(users), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Delete user + all related data
@app.route('/admin/delete_user', methods=['POST'])
def delete_user():
    try:
        data = request.get_json()
        uid = data.get("uid")
        if not uid:
            return jsonify({"error": "Missing UID"}), 400

        # 1️⃣ Delete user from Firebase Auth
        try:
            auth.delete_user(uid)
        except Exception as e:
            print(f"⚠️ Skipping Firebase Auth deletion (possibly already removed): {e}")

        # 2️⃣ Delete user's subcollections (like personal history)
        user_ref = db.collection("users").document(uid)
        try:
            subcollections = user_ref.collections()
            for subcol in subcollections:
                for doc in subcol.stream():
                    doc.reference.delete()
        except Exception as e:
            print(f"⚠️ No subcollections found or failed to delete: {e}")

        # 3️⃣ Delete user's document
        user_ref.delete()

        # 4️⃣ Delete all user entries in global "history" collection
        history_docs = db.collection("history").where("userId", "==", uid).stream()
        deleted_count = 0
        for doc in history_docs:
            doc.reference.delete()
            deleted_count += 1

        return jsonify({
            "message": f"✅ User {uid} and related data deleted successfully.",
            "deleted_history_entries": deleted_count
        }), 200

    except Exception as e:
        print(f"❌ Error deleting user: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ Update user role or name
@app.route('/admin/update_user', methods=['POST'])
def update_user():
    try:
        data = request.get_json()
        uid = data.get("uid")
        name = data.get("name")
        role = data.get("role")

        if not uid:
            return jsonify({"error": "Missing UID"}), 400

        update_data = {}
        if name:
            update_data["name"] = name
        if role:
            update_data["role"] = role

        if not update_data:
            return jsonify({"error": "No data to update"}), 400

        db.collection("users").document(uid).update(update_data)
        return jsonify({"message": "✅ User updated successfully."}), 200

    except Exception as e:
        print(f"❌ Error updating user: {e}")
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------
# 🔹 RUN SERVER
# ------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)