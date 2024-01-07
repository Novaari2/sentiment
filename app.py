from flask import Flask,render_template,request,url_for
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
import re
import string
import sys
import logging
import ssl

app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv('20191002-reviews.csv', encoding='latin-1')
    df = df[['reviewContent', 'rating']].dropna()

    def clean_text(text):
        return re.sub('[^a-zA-Z]',' ',text).lower()
    
    df['cleaned_text'] = df['reviewContent'].apply(lambda x: clean_text(x))
    df['label'] = df['rating'].map({5: 'positif', 4: 'positif', 3: 'netral', 2: 'negatif', 1: 'negatif'})

    def count_punct(review):
        count = sum([1 for char in review if char in string.punctuation])
        return round(count/(len(review) - review.count(" ")), 3)*100
    
    df['review_len'] = df['reviewContent'].apply(lambda x: len(str(x)) - str(x).count(" "))
    df['punct'] = df['reviewContent'].apply(lambda x: count_punct(str(x)))

    def tokenization(text):
        tokenized = text.split()
        return tokenized
    
    df['tokens'] = df['cleaned_text'].apply(lambda x: tokenization(x))

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # lemmatization
    nltk.download('stopwords')
    all_stopwords = stopwords.words('indonesian')
    all_stopwords.remove('tidak')

    nltk.download('wordnet')
    nltk.download('omw-1.4')
    def lemmatize_text(token_list):
        return " ".join([lemmatizer.lemmatize(token) for token in token_list if not token in set(all_stopwords)])
    
    lemmatizer = nltk.stem.WordNetLemmatizer()
    df['lemmatize_review'] = df['tokens'].apply(lambda x: lemmatize_text(x))

    X = df[['lemmatize_review','review_len','punct']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    tfidf = TfidfVectorizer(max_df= 0.5, min_df = 2)
    tfidf_train = tfidf.fit_transform(X_train['lemmatize_review'])
    tfidf_test = tfidf.transform(X_test['lemmatize_review'])

    # prediction
    classifier = SVC(kernel='linear', random_state=10)
    classifier.fit(tfidf_train, y_train)
    classifier.score(tfidf_test, y_test)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tfidf.transform(data).toarray()
        my_prediction = classifier.predict(vect)
    
    return render_template('result.html',prediction = my_prediction)

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

if __name__ == '__main__':
    app.run(debug=True)
    