import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib
import re

print("Loading dataset...")
df = pd.read_csv('news.csv')

# Smart safeguard: Rename columns if the dataset uses different names
col_mapping = {'title': 'text', 'statement': 'text', 'class': 'label'}
df.rename(columns=col_mapping, inplace=True)

# Drop any empty rows to prevent errors
df.dropna(subset=['text', 'label'], inplace=True)

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text)) 
    text = text.lower() 
    return text

print("Cleaning text...")
df['text'] = df['text'].apply(clean_text)

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {round(score*100, 2)}%')

joblib.dump(pac, 'model.pkl')
joblib.dump(tfidf_vectorizer, 'vectorizer.pkl')
print("Success! model.pkl and vectorizer.pkl have been created.")