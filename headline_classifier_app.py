import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Load pipelines once
@st.cache_resource
def load_pipelines():
    emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    return emotion_pipe, sentiment_pipe

emotion_classifier, sentiment_classifier = load_pipelines()

# Extract headlines from url

def extract_headline(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.find("title")
        return title.text.strip() if title else None
    except Exception as e:
        return None


def classify_headline(headline):
    emotion_scores = emotion_classifier(headline)[0]
    top_emotion = max(emotion_scores, key=lambda x: x["score"])
    # Get sentiment result
    sentiment_result = sentiment_classifier(headline)[0]

    # Combine results
    return {
        "headline": headline,
        "emotion": top_emotion['label'],
        "emotion_score": round(top_emotion['score'], 3),
        "sentiment": sentiment_result['label'],
        "sentiment_score": round(sentiment_result['score'], 3)}

# Streamlit UI

st.title("Headline Emotion & Sentiment Analyzer")

url = st.text_input("Paste the URL of a news article:")

if url:
    st.write("Fetching headline...")
    headline = extract_headline(url)

    if headline:
        st.success(f"Extracted Headline: *{headline}*")
        with st.spinner("Classifying..."):
            result = classify_headline(headline)

        st.markdown("### 🎯 Classification Result")
        st.write(f"**Emotion:** {result['emotion']} ({result['emotion_score']})")
        st.write(f"**Sentiment:** {result['sentiment']} ({result['sentiment_score']})")

    else:
        st.error("Could not extract a headline from the page.")
