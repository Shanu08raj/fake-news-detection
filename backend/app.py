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

# Load model and vectorizer
model = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]

    return " ".join(words)


#detect if the input is a factual query for fact-checking mode.
def is_fact_query(text):
    text = text.lower().strip()

    keywords = [
        "is", "are", "was", "were", "has", "have",
        "how many", "what", "who", "when", "where",
        "does", "do", "did"
    ]

    factual_patterns = [
        "states", "capital", "population",
        "country", "president", "prime minister",
        "currency", "continent"
    ]

    return any(text.startswith(k) for k in keywords) or any(word in text for word in factual_patterns)


#function for related news using Tavily API.
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


# Route for prediction.
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    news = data['text']

    # Rule-based obvious fake detection
    fake_keywords = [
        "aliens",
        "breathe underwater",
        "time travel",
        "zombie",
        "dragon",
        "ghost",
        "dinosaurs alive"
    ]

    matched_keywords = [word for word in fake_keywords if word in news.lower()]

    if matched_keywords:
        confidence = min(70 + (len(matched_keywords) * 10), 99)

        return jsonify({
            "prediction": "Fake News",
            "confidence": f"{confidence:.2f}%"
        })

    # Fact-check mode
    if is_fact_query(news):
        articles = search_related_news(news)

        if len(articles) > 0:
            return jsonify({
                "prediction": "Real",
                "confidence": "Web Verified"
            })

    # ML prediction mode
    cleaned_text = clean_text(news)
    vectorized_text = vectorizer.transform([cleaned_text])

    probability = model.predict_proba(vectorized_text)[0]

    fake_prob = probability[0]
    real_prob = probability[1]

    confidence = max(probability) * 100

    # Custom threshold
    if fake_prob > 0.60:
        result = "Fake News"
    else:
        result = "Real News"

    return jsonify({
        "prediction": result,
        "confidence": f"{confidence:.2f}%"
    })


# route for related news.
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

# Run the flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)