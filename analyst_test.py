import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')

import string
import re

import nltk
import ssl
import os
from imblearn.over_sampling import RandomOverSampler
from nltk.corpus import stopwords
import Sastrawi

from wordcloud import WordCloud

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

pd.pandas.set_option('display.max_columns', None)

filename = 'Tokopedia.csv'
df = pd.read_csv(filename, encoding='latin-1')

data = df[['Ulasan', 'Rating']].dropna()

df_copy =df.copy()

# Labelling
label = []
for index, row in df_copy.iterrows():
    if row['Rating'] == 1 or row['Rating'] == 2:
        label.append(0)
    else:
        label.append(1)
df_copy['label'] = label

# Preprocessing
# def casefolding(Review):
#     return Review.lower()

# df_copy['Ulasan'] = df_copy['Ulasan'].apply(casefolding)
df_copy['Ulasan'] = df_copy['Ulasan'].str.lower()

# normalisasi
norm = {"dgn": "dengan", "yg": "yang", "brg": "barang", "tp": "tapi", "bgs": "bagus", "bgt": "banget", "sdh": "sudah", "sdh": "sudah", "bnyk": "banyak", "dg": "dengan", "tlg": "tolong", "gk": "tidak", "sm": "sama"}

def normalisasi(str_text):
    for i in norm:
        str_text = str_text.replace(i, norm[i])
    return str_text

df_copy['Ulasan']= df_copy['Ulasan'].apply(lambda x: normalisasi(x))

# stopword
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
more_stop_words = ['dgn', 'yg', 'brg', 'tp', 'bgs', 'bgt', 'sdh', 'bnyk', 'dg', 'nya', 'gk','oke', 'sm', 'sama']
stop_words = StopWordRemoverFactory().get_stop_words()
new_array = ArrayDictionary(stop_words)
stop_word_remover_new = StopWordRemover(new_array)

def stopword(str_text):
    str_text = stop_word_remover_new.remove(str_text)
    return str_text

df_copy['Ulasan'] = df_copy['Ulasan'].apply(lambda x: stopword(x))

# tokenize
tokenized = df_copy['Ulasan'].apply(lambda x: x.split())

# stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
def stemming(Ulasan):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    do = []
    for w in Ulasan:
        dt = stemmer.stem(w)
        do.append(dt)
    d_clean = []
    d_clean = ' '.join(do)
    print(d_clean)
    return d_clean

tokenized = tokenized.apply(stemming)
tokenized.to_csv('datastemming.csv', index=False)

data_clean = pd.read_csv('datastemming.csv', encoding='latin1')

# menggabungkan 2 attribut
att1 = pd.read_csv('datastemming.csv', encoding='latin1')
att2 = df_copy['label']

result = pd.concat([att1, att2], axis=1)

# penghitungan kata dengan tfidf
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

Ulasan = result['Ulasan']
Ulasan.isnull().sum()
Ulasan.fillna('tidak ada komentar')

cv = CountVectorizer()
term_fit = cv.fit(result['Ulasan'].values.astype('U'))
print(len(term_fit.vocabulary_))

term_fit.vocabulary_

term_freq_all = term_fit.transform(result['Ulasan'].values.astype('U'))
print(term_freq_all)

ulasan_tf = Ulasan[1]
print(ulasan_tf)

term_frequency = term_fit.transform([ulasan_tf])
print(term_frequency)

dokumen = term_fit.transform(result['Ulasan'].values.astype('U'))
tfidf_transformer = TfidfTransformer().fit(dokumen)
print(tfidf_transformer.idf_)

tfidf = tfidf_transformer.transform(term_frequency)
print(tfidf)

# visualisasi
train_s0 = df_copy[df_copy['label'] == 0]
train_s0['Ulasan'] = train_s0['Ulasan'].fillna('tidak ada komentar')

train_s0.head()

from wordcloud import WordCloud
all_text_s0 = ' '.join(word for word in train_s0['Ulasan'])
wordcloud = WordCloud(width=1000, mode='RGBA', background_color = "white", colormap="Reds").generate(all_text_s0)
plt.figure(figsize=[10,8])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

train_s1 = df_copy[df_copy['label'] == 1]
train_s1['Ulasan'] = train_s1['Ulasan'].fillna('tidak ada komentar')
train_s1.head()

all_text_s1 = ' '.join(word for word in train_s1['Ulasan'])
wordcloud = WordCloud(width=1000, height=1000, mode='RGBA', background_color = "white", colormap="Blues").generate(all_text_s1)
plt.figure(figsize=[10,8])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

sentiment_data = pd.value_counts(df_copy['label'], sort = True)
sentiment_data.plot(kind = 'bar', color=['blue', 'red'])
plt.title("Bar Chart")
plt.show()

# split data
result['Ulasan'] = result['Ulasan'].fillna('tidak ada komentar')

# oversampling
# from imblearn.over_sampling import RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# X_resampled, y_resampled = ros.fit_resample(result['Ulasan'].values.astype('U').reshape(-1, 1), result['label'])

# sentiment_data = pd.value_counts(y_resampled, sort = True)
# sentiment_data.plot(kind = 'bar', color=['blue', 'red'])
# plt.title("Bar Chart")
# plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(result['Ulasan'], result['label'], test_size=0.1, stratify=result['label'], random_state=30)
# X_train, X_test, y_train, y_test = train_test_split(X_resampled.flatten(), y_resampled, test_size=0.1, stratify=y_resampled, random_state=30)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(decode_error='replace', encoding='utf-8')

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print(X_train.shape)
print(X_test.shape)

X_train = X_train.toarray()
X_test = X_test.toarray()

# machine learning
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

cv_method = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=999)

params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}

gscv_nb = GridSearchCV(estimator=nb, param_grid=params_NB, cv=cv_method, verbose=1, scoring='accuracy')

gscv_nb.fit(X_train, y_train)
gscv_nb.best_params_

nb = GaussianNB(var_smoothing=0.0015199110829529332)

nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, RocCurveDisplay

print('--------- confusion matrix ---------')

print(confusion_matrix(y_test, y_pred_nb))

print('--------- classification report ---------')

print(classification_report(y_test, y_pred_nb))

RocCurveDisplay.from_estimator(nb, X_test, y_test)

# save model
# import joblib

# joblib.dump(nb, 'sentiment_model.pkl')

new_review = "Pengiriman tidak mengecewakan, selalu tepat waktu"
new_review = new_review.lower()
new_review = normalisasi(new_review)
new_review = stopword(new_review)
new_review = stemming(new_review)
print(new_review)

new_review_tfidf = vectorizer.transform([new_review]).toarray()

sentiment_prediction = nb.predict(new_review_tfidf)
if sentiment_prediction == 1:
    print("Ulasan positif!")
else:
    print("Ulasan negatif.")

