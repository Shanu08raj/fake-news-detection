from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import requests
from dotenv import load_dotenv

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


app = Flask(__name__)
CORS(app)

model = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]

    return " ".join(words)

def search_related_news(query):
    url = "https://api.tavily.com/search"

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": 3
    }

    response = requests.post(url, json=payload)
    data = response.json()

    return data.get("results", [])


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    news = data['text']

    cleaned_text = clean_text(news)
    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0]

    confidence = max(probability) * 100

    result = "Fake News" if prediction == 1 else "Real News"

    return jsonify({
        "prediction": result,
        "confidence": f"{confidence:.2f}%"
    })

@app.route('/related-news', methods=['POST'])
def related_news():
    data = request.json
    news = data['text']

    query = " ".join(news.split()[:10])

    articles = search_related_news(query)

    formatted_articles = []
    for article in articles:
        formatted_articles.append({
            "title": article.get("title"),
            "url": article.get("url")
        })

    return jsonify({
        "articles": formatted_articles
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)