# Text Emotion Classification Using NLP & Machine Learning

A beautiful web application powered by Machine Learning and Natural Language Processing to classify the emotion behind a given text. It can detect emotions like Sadness 😔, Anger 😠, Love ❤️, Surprise 😲, Fear 😨, and Joy 😊.

## Features
- **Machine Learning**: Uses a Logistic Regression model trained on TF-IDF features with 86.3% accuracy.
- **Beautiful UI**: Built with HTML, CSS, and vanilla JS, featuring dynamic floating emojis and glassmorphism.
- **Real-time API**: Powered by a lightweight Flask backend providing predictions instantly.

## How to run locally

1. Install dependencies:
   ```bash
   pip install flask pandas scikit-learn numpy joblib nltk
   ```
2. (Optional) If you want to retrain the model, run:
   ```bash
   python train_model.py
   ```
3. Start the Flask application:
   ```bash
   python app.py
   ```
4. Open your browser and go to `http://localhost:5000`
