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
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

app = Flask(__name__, template_folder='template')

# local
# app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), '/Users/novaariyanto/Development/python/sentiment')
# docker
# app.config['UPLOAD_FOLDER'] = '/app/Tokopedia.csv'

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


# @app.route('/getscrap')
# def scrap():
#     return render_template('scrapping.html')

# @app.route('/scrapping',methods=['POST'])
# def scrapper():
#     if request.method == 'POST':
#         url = request.form['url']
#         if url:
#             options = webdriver.ChromeOptions()
#             options.add_argument("--start-maximized")
#             driver = webdriver.Chrome(options=options)
#             driver.get(url)

#             data = []
#             for i in range(0, 10):
#                 soup = BeautifulSoup(driver.page_source, "html.parser")
#                 containers = soup.findAll('article', attrs={'class': 'css-ccpe8t'})

#                 for container in containers:
#                     try:
#                         review = container.find('span', attrs={'data-testid': 'lblItemUlasan'}).text
#                         rating = container.find('div', attrs={'data-testid': 'icnStarRating'})['aria-label'] if container.find('div', attrs={'data-testid': 'icnStarRating'}) else "N/A"
#                         rating_mapping = {"bintang 1": 1, "bintang 2": 2, "bintang 3": 3, "bintang 4": 4, "bintang 5": 5}
#                         rating = rating_mapping.get(rating, "N/A")

#                         data.append(
#                             (review, rating)
#                         )
#                     except AttributeError:
#                         continue
                
#                 time.sleep(2)
#                 driver.find_element(By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']").click()
#                 time.sleep(3)
            
#             df = pd.DataFrame(data, columns=['Ulasan', 'Rating'])
#             df.to_csv('Tokopedia.csv', index=False)
    
#     return render_template('result_scrapping.html', STATUSCODE=200)


@app.route('/getanalyst')
def analyst():
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

    data_negative = df[ (df['Rating'] == 1) | (df['Rating'] == 2)]
    data_neutral = df[ (df['Rating'] == 3)]
    data_positive = df[ (df['Rating'] == 4) | (df['Rating'] == 5)]

    negative_list = data_negative['lemmatize_review'].tolist()
    neutral_list = data_neutral['lemmatize_review'].tolist()
    positive_list = data_positive['lemmatize_review'].tolist()

    filtered_negative = ("").join(negative_list)
    filtered_negative = filtered_negative.lower()

    filtered_neutral = ("").join(neutral_list)
    filtered_neutral = filtered_neutral.lower()

    filtered_positive = ("").join(positive_list)
    filtered_positive = filtered_positive.lower()

    # wordcloud = WordCloud(max_font_size = 160, margin=0, background_color = "white", colormap="Greens").generate(filtered_positive)
    # plt.figure(figsize=[10,8])
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.margins(x=0, y=0)
    # plt.title("Positive Review Word Cloud")
    # plt.show()

    # wordcloud = WordCloud(max_font_size = 160, margin=0, background_color = "white", colormap="Reds").generate(filtered_negative)
    # plt.figure(figsize=[10,8])
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.margins(x=0, y=0)
    # plt.title("Negative Review Word Cloud")
    # plt.show()

    # if len(filtered_neutral.split()) > 0:
    #     wordcloud = WordCloud(max_font_size = 160, margin=0, background_color = "white", colormap="Blues").generate(filtered_neutral)
    #     plt.figure(figsize=[10,8])
    #     plt.imshow(wordcloud, interpolation='bilinear')
    #     plt.axis("off")
    #     plt.margins(x=0, y=0)
    #     plt.title("Neutral Review Word Cloud")
    #     plt.show()
    # else:
    #     print("Teks tidak mengandung kata. Tidak dapat membuat Word Cloud.")

    X = df[['lemmatize_review','review_len','punct']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    vectorizer = TfidfVectorizer(decode_error='replace', encoding='utf-8')
    X_train_vect = vectorizer.fit_transform(X_train['lemmatize_review'])
    X_test_vect = vectorizer.transform(X_test['lemmatize_review'])

    X_train_vect = X_train_vect.toarray()
    X_test_vect = X_test_vect.toarray()

    # SVM
    svm_classifier = SVC(kernel='linear', random_state=0)
    svm_classifier.fit(X_train_vect, y_train)
    svm_pred = svm_classifier.predict(X_test_vect)
    accuracy = accuracy_score(y_test, svm_pred)
    report = classification_report(y_test, svm_pred)
    print("Accuracy: ", accuracy_score(y_test, svm_pred))
    print(classification_report(y_test, svm_pred))

    import matplotlib
    matplotlib.use('Agg')
    class_label = ['NEGATIF', 'NETRAL', 'POSITIF']
    data_cm = pd.DataFrame(confusion_matrix(y_test, svm_pred, labels=class_label), index = class_label, columns = class_label)
    sns.heatmap(data_cm, annot = True, fmt = "d")
    plt.title("Confusion Matrix SVM")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    # plt.show()
    plt.savefig('sample_plot.jpg') 


    return render_template('analyst.html', accuracy=accuracy, classification_report=report, plot_image='sample_plot.png')


@app.route('/data_upload')
def data_upload():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                # local
                # file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                # file.save(file_path)

                upload_folder = '/app'
                file_path = os.path.join(upload_folder, file.filename)
                file.save(file_path)

                # Memberikan notifikasi bootstrap success
                response = {'status': 'success', 'message': 'File berhasil diupload'}
            except Exception as e:
                # Jika terjadi kesalahan saat menyimpan file
                response = {'status': 'error', 'message': str(e)}
        else:
            # Jika tidak ada file yang diunggah
            response = {'status': 'error', 'message': 'Tidak ada file yang diunggah'}
    else:
        # Jika bukan metode POST
        response = {'status': 'error', 'message': 'Metode tidak diizinkan'}

    return jsonify(response)


app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

if __name__ == '__main__':
    app.run(debug=True)
    