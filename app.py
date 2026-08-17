import string
import joblib
from flask import Flask, render_template, request, jsonify

# Re-declare text preprocessing functions so that they match exactly what the model expects
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

from nltk.corpus import stopwords
import nltk
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

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

app = Flask(__name__)

# Load models and mappings
model = None
vectorizer = None
emotions_map = None

try:
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    emotions_map = joblib.load('emotions_map.pkl')
    print("Successfully loaded model and artifacts.")
except Exception as e:
    print(f"Error loading models. Did you run train_model.py? Error: {e}")

# Emotion mappings to emojis for UI richness
emotion_emojis = {
    'sadness': '😔',
    'anger': '😠',
    'love': '❤️',
    'surprise': '😲',
    'fear': '😨',
    'joy': '😊'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer or not emotions_map:
        return jsonify({'error': 'Model is not loaded. Please wait for training to complete.'}), 500
        
    data = request.json
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'error': 'Please enter some text.'}), 400
        
    # Preprocess text
    cleaned_text = preprocess(text)
    if not cleaned_text.strip():
        # Text became empty after preprocessing (e.g. only stopwords)
        return jsonify({'error': 'Text is too short or lacks meaningful words after preprocessing.'}), 400
        
    # Predict
    features = vectorizer.transform([cleaned_text])
    prediction_num = model.predict(features)[0]
    
    emotion_name = emotions_map.get(prediction_num, "Unknown")
    emoji = emotion_emojis.get(emotion_name, '😐')
    
    # Get probabilities for all classes
    probabilities = model.predict_proba(features)[0]
    confidences = {}
    for i, prob in enumerate(probabilities):
        confidences[emotions_map[i]] = round(prob * 100, 2)
        
    return jsonify({
        'emotion': emotion_name,
        'emoji': emoji,
        'confidence': confidences[emotion_name],
        'all_confidences': confidences,
        'cleaned_text': cleaned_text
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
