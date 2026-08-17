import streamlit as st
import string
import joblib
import pandas as pd
from nltk.corpus import stopwords
import nltk

# Streamlit page config must be the first Streamlit command
st.set_page_config(
    page_title="Text Emotion Classification",
    page_icon="✨",
    layout="centered"
)

# Function to ensure stopwords are downloaded
@st.cache_resource
def get_stopwords():
    try:
        return set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        return set(stopwords.words('english'))

stop_words = get_stopwords()

# Preprocessing functions
def remove_punc(txt):
    return txt.translate(str.maketrans('','',string.punctuation))

def remove_numbers(txt):
    new = ""
    for i in txt:
        if not i.isdigit():
            new = new + i
    return new

def remove_emojis(txt):
    new = ""
    for i in txt:
        if i.isascii():
            new += i
    return new

def remove_stopwords(txt):
    words = txt.split()
    cleaned = [i for i in words if not i in stop_words]
    return ' '.join(cleaned)

def preprocess(txt):
    txt = txt.lower()
    txt = remove_punc(txt)
    txt = remove_numbers(txt)
    txt = remove_emojis(txt)
    txt = remove_stopwords(txt)
    return txt

# Load models and mappings
@st.cache_resource
def load_models():
    try:
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        emotions_map = joblib.load('emotions_map.pkl')
        return model, vectorizer, emotions_map
    except Exception as e:
        return None, None, None

model, vectorizer, emotions_map = load_models()

emotion_emojis = {
    'sadness': '😔',
    'anger': '😠',
    'love': '❤️',
    'surprise': '😲',
    'fear': '😨',
    'joy': '😊'
}

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    h1 {
        background: linear-gradient(to right, #fff, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    .stTextArea textarea {
        background-color: rgba(30, 41, 59, 0.7) !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    div[data-testid="stButton"] button {
        background-color: #6366f1 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        width: 100% !important;
        font-weight: 600 !important;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #4f46e5 !important;
    }
    .result-container {
        text-align: center;
        padding: 2rem;
        background: rgba(30, 41, 59, 0.7);
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        margin-top: 1rem;
    }
    .result-emoji {
        font-size: 5rem;
        animation: bounce 2s infinite;
    }
    .emotion-title {
        font-size: 2rem;
        font-weight: bold;
        margin: 10px 0;
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
</style>
""", unsafe_allow_html=True)

st.title("✨ Text Emotion Classification")
st.markdown("<p style='text-align: center; color: #94a3b8;'>Using NLP & Machine Learning to understand feelings</p>", unsafe_allow_html=True)

if not model:
    st.error("Model files not found! Please run the training script or Jupyter Notebook to generate them.")
    st.stop()

# Input area
user_input = st.text_area("Type or paste your text here...", height=150)

if st.button("Analyze Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            cleaned_text = preprocess(user_input)
            
            if not cleaned_text.strip():
                st.error("Text is too short or lacks meaningful words after preprocessing.")
            else:
                features = vectorizer.transform([cleaned_text])
                prediction_num = model.predict(features)[0]
                emotion_name = emotions_map.get(prediction_num, "Unknown")
                emoji = emotion_emojis.get(emotion_name, '😐')
                
                probabilities = model.predict_proba(features)[0]
                confidences = {emotions_map[i]: round(prob * 100, 2) for i, prob in enumerate(probabilities)}
                
                # Display Result
                st.markdown(f"""
                <div class="result-container">
                    <div class="result-emoji">{emoji}</div>
                    <div class="emotion-title">{emotion_name.capitalize()}</div>
                    <p style="color: #94a3b8; font-size: 1.1rem;">Confidence: <strong>{confidences[emotion_name]}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### Detailed Breakdown")
                
                sorted_emotions = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
                
                for em, conf in sorted_emotions:
                    if em != emotion_name:
                        st.write(f"**{emotion_emojis.get(em, '😐')} {em.capitalize()}**: {conf}%")
                        st.progress(int(conf))
