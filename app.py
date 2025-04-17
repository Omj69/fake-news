from flask import Flask, request, render_template
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Flask App
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    predictions = []
    if request.method == 'POST':
        news_text = request.form['news']
        news_list = [line.strip() for line in news_text.strip().split('\n') if line.strip()]
        processed = [preprocess(news) for news in news_list]
        vec = vectorizer.transform(processed)
        preds = model.predict(vec)

        predictions = list(zip(news_list, preds))

    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
