import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample
import pickle

# --- Step 1: Load Dataset ---
df = pd.read_csv("go_emotions_dataset.csv")

# --- Step 2: Identify Emotion Columns Dynamically ---
emotion_columns = df.columns[3:]  # emotion labels start from 4th column
texts = df["text"]

# Keep only rows that have at least one emotion label
valid_rows = df[emotion_columns].sum(axis=1) > 0
df = df[valid_rows]

# Extract emotion names directly (not numeric indices)
emotion_labels = df[emotion_columns].idxmax(axis=1)

# --- Step 3: Combine into one clean DataFrame ---
data = pd.DataFrame({
    "text": df["text"],
    "emotion": emotion_labels
}).dropna()

print(f"✅ Loaded {len(data)} samples with {data['emotion'].nunique()} unique emotions.")
print("🎭 Sample emotions:", data["emotion"].unique()[:10])

# --- Step 4: Text Cleaning Function ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["text"] = data["text"].apply(clean_text)

# --- Step 5: Handle Class Imbalance (Upsampling) ---
max_size = data["emotion"].value_counts().max()
balanced_data = pd.concat([
    resample(
        data[data["emotion"] == emotion],
        replace=True,
        n_samples=max_size,
        random_state=42
    )
    for emotion in data["emotion"].unique()
])

balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

print("✅ Dataset balanced successfully.")
print(balanced_data["emotion"].value_counts().head())

# --- Step 6: TF-IDF Vectorization ---
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(balanced_data["text"])

# --- Step 7: Encode Labels ---
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(balanced_data["emotion"])

# --- Step 8: Train Naive Bayes Model ---
model = MultinomialNB(alpha=0.5)
model.fit(X, y)

# --- Step 9: Save Model, Vectorizer, and Label Encoder ---
pickle.dump(model, open("emotion_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

print("✅ Model retrained and saved successfully.")

# --- Step 10: Test with Sample Sentences ---
test_sentences = [
    "I am so happy today!",
    "This is the worst day ever.",
    "I feel really sad and alone.",
    "You are such a wonderful friend.",
    "I am nervous about tomorrow's exam.",
    "I can’t believe this happened!",
    "Everything feels peaceful now."
]

X_test = vectorizer.transform([clean_text(s) for s in test_sentences])
y_pred = label_encoder.inverse_transform(model.predict(X_test))

print("\n🎯 Sample Predictions:")
for sentence, emotion in zip(test_sentences, y_pred):
    print(f"→ {sentence}  →  {emotion}")
