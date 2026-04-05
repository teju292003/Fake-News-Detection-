from django.shortcuts import render
import joblib
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'detector', 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'detector', 'vectorizer.pkl')

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    model = None
    vectorizer = None
    print(f"Error loading models: {e}")

def clean_input(text):
    text = re.sub(r'[^\w\s]', '', str(text))
    return text.lower()

def home(request):
    prediction = None
    news_text = ""

    if request.method == 'POST':
        news_text = request.POST.get('news_text', '')
        
        if news_text and model and vectorizer:
            cleaned_text = clean_input(news_text)
            vectorized_text = vectorizer.transform([cleaned_text])
            
            pred = model.predict(vectorized_text)[0]
            
            if str(pred).upper() in ['FAKE', '0']:
                prediction = "FAKE NEWS"
            else:
                prediction = "REAL NEWS"

    return render(request, 'detector/index.html', {
        'prediction': prediction,
        'news_text': news_text,
        'model_loaded': model is not None
    })