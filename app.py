import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Json dosyasını okuma
with open("data.json", "r", encoding="UTF-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)


stop_words = set(stopwords.words('turkish'))
# Metin ön işleme
def filter(text):
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("<br />"," ")
    text = text.strip()
    return text

# Stopwords kaldırma
def remove_stopwords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Metin verileri ve duygu etiketleri
X = df['Görüş']
y = df['Durum']

# Veri ön işleme
X = X.apply(filter)
X = X.apply(remove_stopwords)

# Veriyi eğitim ve test verilerine ayırma 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Metin verilerini TF-IDF özellikleri olarak dönüştürme
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Naive Bayes sınıflandırma modelini eğitme
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tfidf, y_train)

# Modeli kullanarak tahminler yapma
y_pred = naive_bayes.predict(X_test_tfidf)

# Modelin doğruluğunu değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print("Model Doğruluğu:", accuracy)

# Sınıflandırma raporunu yazdırma
print(classification_report(y_test, y_pred))

# Yeni metinlerin duygusunu tahmin etme
print("Yeni metinlerin duygusunu tahmin edin: ")
metin1 = input("Metin 1: ")
metin2 = input("Metin 2: ")
print()

new_texts = [metin1, metin2]
new_texts_tfidf = tfidf_vectorizer.transform(new_texts)
new_predictions = naive_bayes.predict(new_texts_tfidf)

for i, text in enumerate(new_texts):
    print(f"Metin: {text}")
    print(f"Duygu: {new_predictions[i]}")
    print()
