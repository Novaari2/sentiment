from flask import Flask,render_template,request,url_for,jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
import re, time, string, sys, logging, os, ssl
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv('Tokopedia.csv', encoding='latin-1')
    df = df[['Ulasan', 'Rating']].dropna()

    def clean_text(text):
        return re.sub('[^a-zA-Z]',' ',text).lower()
    
    df['cleaned_text'] = df['Ulasan'].apply(lambda x: clean_text(x)) 
    df['label'] = df['Rating'].map({5: "POSITIF", 4: "POSITIF", 3: "NETRAL", 2: "NEGATIF", 1: "NEGATIF"})

    def count_punct(review):
        count = sum([1 for char in review if char in string.punctuation])
        return round(count/(len(review) - review.count(" ")), 3)*100
    
    df['review_len'] = df['Ulasan'].apply(lambda x: len(str(x)) - str(x).count(" "))
    df['punct'] = df['Ulasan'].apply(lambda x: count_punct(str(x)))

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

    nltk_download = 'nltk_download'


    if not os.path.exists(nltk_download):
        os.mkdir(nltk_download)

    if not os.path.exists(os.path.join(nltk_download, 'corpora/stopwords')):
        nltk.download('stopwords', download_dir=nltk_download)

    if not os.path.exists(os.path.join(nltk_download, 'corpora/wordnet.zip')):
        nltk.download('wordnet', download_dir=nltk_download)

    if not os.path.exists(os.path.join(nltk_download, 'corpora/omw-1.4.zip')):
        nltk.download('omw-1.4', download_dir=nltk_download)

    nltk.data.path.append(nltk_download)

    all_stopwords = stopwords.words('indonesian')
    custom_stopwords = {'tidak', 'dgn', 'yg', 'brg', 'tp', 'bgs', 'bgt', 'sdh', 'bnyk', 'dg', 'nya', 'gk', 'oke', 'sm', 'sama'}
    for word in custom_stopwords:
        if word not in all_stopwords:
            all_stopwords.append(word)
    all_stopwords = set(all_stopwords)

    # lemmatization
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


@app.route('/getscrap')
def scrap():
    return render_template('scrapping.html')

@app.route('/scrapping',methods=['POST'])
def scrapper():
    if request.method == 'POST':
        url = request.form['url']
        if url:
            options = webdriver.ChromeOptions()
            options.add_argument("--start-maximized")
            driver = webdriver.Chrome(options=options)
            driver.get(url)

            data = []
            for i in range(0, 1):
                soup = BeautifulSoup(driver.page_source, "html.parser")
                containers = soup.findAll('article', attrs={'class': 'css-ccpe8t'})

                for container in containers:
                    try:
                        review = container.find('span', attrs={'data-testid': 'lblItemUlasan'}).text
                        rating = container.find('div', attrs={'data-testid': 'icnStarRating'})['aria-label'] if container.find('div', attrs={'data-testid': 'icnStarRating'}) else "N/A"
                        rating_mapping = {"bintang 1": 1, "bintang 2": 2, "bintang 3": 3, "bintang 4": 4, "bintang 5": 5}
                        rating = rating_mapping.get(rating, "N/A")

                        data.append(
                            (review, rating)
                        )
                    except AttributeError:
                        continue
                
                time.sleep(2)
                driver.find_element(By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']").click()
                time.sleep(3)
            
            df = pd.DataFrame(data, columns=['Ulasan', 'Rating'])
            df.to_csv('Tokopedia.csv', index=False)
    
    return render_template('result_scrapping.html', STATUSCODE=200)


app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

if __name__ == '__main__':
    app.run(debug=True)
    