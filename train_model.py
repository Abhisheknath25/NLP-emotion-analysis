import pandas as pd
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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

# Stopwords are downloaded in the script to ensure they are available
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
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

print("Loading data...")
df = pd.read_csv('train.txt', sep=';', header=None, names=['text', 'emotion'])

unique_emotions = df['emotion'].unique()
emotion_to_num = {emo: i for i, emo in enumerate(unique_emotions)}
num_to_emotion = {i: emo for emo, i in emotion_to_num.items()}
df['emotion'] = df['emotion'].map(emotion_to_num)

print("Preprocessing text...")
df['text'] = df['text'].apply(preprocess)

print("Training vectorizer...")
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df['text'])
y = df['emotion']

print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_tfidf, y)

print("Saving model and artifacts...")
joblib.dump(model, 'model.pkl')
joblib.dump(tfidf_vectorizer, 'vectorizer.pkl')
joblib.dump(num_to_emotion, 'emotions_map.pkl')

print("Done!")
