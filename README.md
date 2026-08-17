# Text Emotion Classification Using NLP & Machine Learning

A beautiful web application powered by Machine Learning and Natural Language Processing to classify the emotion behind a given text. It can detect emotions like Sadness 😔, Anger 😠, Love ❤️, Surprise 😲, Fear 😨, and Joy 😊.

## Features
- **Machine Learning**: Uses a Logistic Regression model trained on TF-IDF features with 86.3% accuracy.
- **Beautiful UI**: Built entirely in Python using **Streamlit**, featuring custom dark-mode CSS and emoji breakdowns.
- **Instant Predictions**: Runs locally in your browser with real-time text analysis.

## How to run locally

1. Install dependencies:
   ```bash
   pip install streamlit pandas scikit-learn numpy joblib nltk
   ```
2. Start the Streamlit application:
   ```bash
   python -m streamlit run app.py
   ```
3. Open your browser and go to: [http://localhost:8501](http://localhost:8501)
