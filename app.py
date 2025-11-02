from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import json
import firebase_admin
from firebase_admin import credentials, auth, firestore

# ------------------------------------------------------------
# 🔹 Initialize Flask App
# ------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------
# 🔹 Initialize Firebase Admin SDK
# ------------------------------------------------------------
try:
    if "FIREBASE_CREDENTIALS" in os.environ:
        cred_dict = json.loads(os.environ["FIREBASE_CREDENTIALS"])
        cred = credentials.Certificate(cred_dict)
    else:
        cred = credentials.Certificate("firebase_admin_key.json")

    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase initialized successfully.")
except Exception as e:
    print(f"⚠️ Firebase initialization failed: {e}")
    db = None

# ------------------------------------------------------------
# 🔹 Load Machine Learning Model
# ------------------------------------------------------------
try:
    model = pickle.load(open("emotion_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    print("✅ ML Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading ML models: {e}")
    model = vectorizer = label_encoder = None

# ------------------------------------------------------------
# 🔹 Base Route
# ------------------------------------------------------------
@app.route('/')
def home():
    return jsonify({"message": "Emotion Recognition API is running!"})

# ------------------------------------------------------------
# 🔹 Predict Emotion
# ------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

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
        }), 200

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": "Model error"}), 500

# ------------------------------------------------------------
# 🔹 ADMIN FEATURES
# ------------------------------------------------------------

# ✅ Fetch all users
@app.route('/admin/get_users', methods=['GET'])
def get_users():
    try:
        users = []
        docs = db.collection("users").get()
        for doc in docs:
            user_data = doc.to_dict()
            user_data["uid"] = doc.id
            users.append(user_data)
        return jsonify(users), 200
    except Exception as e:
        print(f"❌ Error fetching users: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ Delete user (Firebase Auth + Firestore + global history)
@app.route('/admin/delete_user', methods=['POST'])
def delete_user():
    try:
        data = request.get_json()
        uid = data.get("uid")
        if not uid:
            return jsonify({"error": "Missing UID"}), 400

        # 1️⃣ Delete from Firebase Authentication
        try:
            auth.delete_user(uid)
            print(f"🗑️ Firebase Auth: Deleted user {uid}")
        except Exception as e:
            print(f"⚠️ Auth deletion skipped: {e}")

        # 2️⃣ Delete user’s subcollections
        user_ref = db.collection("users").document(uid)
        try:
            for subcol in user_ref.collections():
                for doc in subcol.stream():
                    doc.reference.delete()
        except Exception as e:
            print(f"⚠️ Failed to delete subcollections for {uid}: {e}")

        # 3️⃣ Delete Firestore document
        user_ref.delete()

        # 4️⃣ Delete from global history
        deleted_count = 0
        history_docs = db.collection("history").where("userId", "==", uid).stream()
        for doc in history_docs:
            doc.reference.delete()
            deleted_count += 1

        print(f"✅ Deleted {uid} with {deleted_count} related history records.")
        return jsonify({
            "message": f"User {uid} deleted successfully.",
            "deleted_history_entries": deleted_count
        }), 200

    except Exception as e:
        print(f"❌ Error deleting user: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ Update user (role or name)
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
        print(f"✅ Updated user {uid} with data: {update_data}")
        return jsonify({"message": "User updated successfully."}), 200

    except Exception as e:
        print(f"❌ Error updating user: {e}")
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------
# 🔹 Run Flask App
# ------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)