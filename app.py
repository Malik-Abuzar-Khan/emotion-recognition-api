from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials, auth, firestore

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)

# ✅ Initialize Firebase Admin SDK
try:
    cred = credentials.Certificate("firebase_admin_key.json")  # <-- update path if needed
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
# 🔹 API ROUTES
# ------------------------------------------------------------

@app.route('/')
def home():
    return jsonify({"message": "Emotion Recognition API is running!"})

# 🔹 Predict Emotion
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
# 🔹 ADMIN FEATURES (MANAGE USERS)
# ------------------------------------------------------------

# ✅ Get all users (from Firebase Auth + Firestore)
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


# ✅ Delete a user completely (Firestore + Firebase Auth)
@app.route('/admin/delete_user', methods=['POST'])
def delete_user():
    try:
        data = request.get_json()
        uid = data.get("uid")
        if not uid:
            return jsonify({"error": "Missing UID"}), 400

        # 🔹 1. Delete user from Firebase Authentication
        try:
            auth.delete_user(uid)
        except Exception as e:
            print(f"⚠️ Firebase Auth deletion failed (maybe user not found): {e}")

        # 🔹 2. Delete user's Firestore document
        user_ref = db.collection("users").document(uid)
        user_ref.delete()

        # 🔹 3. Delete user's subcollection (history)
        history_ref = user_ref.collection("history")
        histories = history_ref.stream()
        for doc in histories:
            doc.reference.delete()

        # 🔹 4. Delete related entries in global "history" collection
        all_history = db.collection("history").where("userId", "==", uid).stream()
        for doc in all_history:
            doc.reference.delete()

        return jsonify({"message": f"User {uid} and all related data deleted successfully!"}), 200

    except Exception as e:
        print(f"❌ Error deleting user: {e}")
        return jsonify({"error": str(e)}), 500


# ✅ Update user details (name or role)
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

        if update_data:
            db.collection("users").document(uid).update(update_data)
            return jsonify({"message": "User updated successfully"}), 200
        else:
            return jsonify({"error": "No data to update"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# 🔹 RUN SERVER
# ------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)