import streamlit as st
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import matplotlib.pyplot as plt

MAX_LEN = 200

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lstm_model.h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def prepare_input(text, tokenizer):
    text = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    return padded

def extract_sentiment_phrases(text):
    positive_keywords = ["great", "excellent", "amazing", "powerful", "brilliant", "emotional", "beautiful", "touching", "fantastic", "love", "stunning", "well done", "enjoyed"]
    negative_keywords = ["boring", "bad", "weak", "predictable", "rushed", "disappointing", "waste", "poor", "underdeveloped", "dragged", "terrible", "flat"]

    sentences = re.split(r'[.!?]', text)
    positive = []
    negative = []

    for sentence in sentences:
        lowered = sentence.lower()
        if any(word in lowered for word in positive_keywords):
            positive.append(sentence.strip())
        elif any(word in lowered for word in negative_keywords):
            negative.append(sentence.strip())
    
    return positive, negative

st.set_page_config(page_title="🎬 IMDB Sentiment Analyzer", layout="centered")

st.title("🎬 IMDB Sentiment Classifier")
st.markdown("Write a review below and see whether it's **Positive** or **Negative** using a pre-trained LSTM model.")

user_input = st.text_area("📝 Enter your movie review here:")

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        model = load_model()
        tokenizer = load_tokenizer()
        prepared_input = prepare_input(user_input, tokenizer)

        prediction = model.predict(prepared_input)[0][0]
        label = "🌟 Positive" if prediction > 0.5 else "👎 Negative"
        confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)

        st.markdown(f"## Prediction: {label}")
        st.markdown(f"### Confidence: **{confidence:.2%}**")

        st.progress(confidence)

        fig, ax = plt.subplots()
        ax.pie([confidence, 1 - confidence],
            labels=["Positive", "Negative"] if prediction > 0.5 else ["Negative", "Positive"],
            autopct="%1.1f%%",
            colors=["#00cc99", "#ff4d4d"],
            startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("🟢 Positive Phrases")
        pos, neg = extract_sentiment_phrases(user_input)
        if pos:
            for p in pos:
                st.markdown(f"- ✅ {p}")
        else:
            st.write("No clearly positive phrases detected.")

        st.subheader("🔴 Negative Phrases")
        if neg:
            for n in neg:
                st.markdown(f"- ❌ {n}")
        else:
            st.write("No clearly negative phrases detected.")
