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
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords

from wordcloud import WordCloud

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

pd.pandas.set_option('display.max_columns', None)

filename = '20191002-reviews.csv'
df = pd.read_csv(filename, encoding='latin-1')
df.head()

data = df[['reviewContent', 'rating']].dropna()
data.head()

def clean_text(text):
    return re.sub('[^a-zA-Z]',' ',text).lower()
data['clean_text'] = data['reviewContent'].apply(lambda x: clean_text(x))
data['label'] = data['rating'].map({5: 'positif', 4: 'positif', 3: 'netral', 2: 'negatif', 1: 'negatif'})

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100
data['review_len'] = data['reviewContent'].apply(lambda x: len(x) - x.count(" "))
data['punct'] = data['reviewContent'].apply(lambda x: count_punct(x))
data.head()

def tokenize_text(text):
    tokenize_text = text.split()
    return tokenize_text
data['tokens']  =   data['clean_text'].apply(lambda x: tokenize_text(x))
data.head()

all_stopwords = stopwords.words('indonesian')
all_stopwords.remove('tidak')

def lemmatize_text(token_list):
    return " ".join([lemmatizer.lemmatize(token) for token in token_list if not token in set(all_stopwords)])

lemmatizer = nltk.stem.WordNetLemmatizer()
data['lemmatize_review'] = data['tokens'].apply(lambda x: lemmatize_text(x))
data.head()

print(f"Input data has {len(data)} rows and {len(data.columns)} columns")
print(f"rating 1 = {len(data[data['rating'] == 1])} rows")
print(f"rating 2 = {len(data[data['rating'] == 2])} rows")
print(f"rating 3 = {len(data[data['rating'] == 3])} rows")
print(f"rating 4 = {len(data[data['rating'] == 4])} rows")
print(f"rating 5 = {len(data[data['rating'] == 5])} rows")

print(f"Number of null in label: {data['rating'].isnull().sum()}")
print(f"Number of null in label: {data['reviewContent'].isnull().sum()}")
sns.countplot(x='rating', data=data)

data_negative = data[ (data['rating'] == 1) | (data['rating'] == 2)]
data_neutral = data[ (data['rating'] == 3)]
data_positive = data[ (data['rating'] == 4) | (data['rating'] == 5)]

negative_list = data_negative['lemmatize_review'].tolist()
neutral_list = data_neutral['lemmatize_review'].tolist()
positive_list = data_positive['lemmatize_review'].tolist()

filtered_negative = ("").join(negative_list)
filtered_negative = filtered_negative.lower()

filtered_neutral = ("").join(neutral_list)
filtered_neutral = filtered_neutral.lower()

filtered_positive = ("").join(positive_list)
filtered_positive = filtered_positive.lower()

wordcloud = WordCloud(max_font_size = 160, margin=0, background_color = "white", colormap="Greens").generate(filtered_positive)
plt.figure(figsize=[10,8])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.title("Positive Review Word Cloud")
plt.show()

wordcloud = WordCloud(max_font_size = 160, margin=0, background_color = "white", colormap="Reds").generate(filtered_negative)
plt.figure(figsize=[10,8])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.title("Negative Review Word Cloud")
plt.show()

wordcloud = WordCloud(max_font_size = 160, margin=0, background_color = "white", colormap="Blues").generate(filtered_neutral)
plt.figure(figsize=[10,8])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.title("Neutral Review Word Cloud")
plt.show()

data['lemmatize_review'] = data['lemmatize_review'].astype(str)
X = data[['lemmatize_review','review_len','punct']]
y = data['label']
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_df= 0.5, min_df = 2)
tfidf_train = tfidf.fit_transform(X_train['lemmatize_review'])
tfidf_test = tfidf.transform(X_test['lemmatize_review'])

X_train_vect = pd.DataFrame(tfidf_train.toarray(), columns=tfidf.get_feature_names_out())
X_test_vect = pd.DataFrame(tfidf_test.toarray(), columns=tfidf.get_feature_names_out())

X_train_combined = pd.concat([X_train[['review_len','punct']].reset_index(drop=True),
               X_train_vect], axis=1)
X_test_combined = pd.concat([X_test[['review_len','punct']].reset_index(drop=True), X_test_vect], axis=1)

# X_train_vect.head()

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train_vect, y_train)
naive_bayes_pred = classifier.predict(X_test_vect)

print(classification_report(y_test, naive_bayes_pred))

class_label = ['negative', 'neutral', 'positive']
data_cm = pd.DataFrame(confusion_matrix(y_test, naive_bayes_pred), index = class_label, columns = class_label)
sns.heatmap(data_cm, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# # Random Forest Classifier
# from sklearn.ensemble import RandomForestClassifier

# classifier = RandomForestClassifier(n_estimators = 150)
# classifier.fit(X_train_vect, y_train)
# random_forest_pred = classifier.predict(X_test_vect)

# print(classification_report(y_test, random_forest_pred))

# # confusion matrix
# class_label = ['negative', 'neutral', 'positive']
# data_cm = pd.DataFrame(confusion_matrix(y_test, random_forest_pred), index = class_label, columns = class_label)
# sns.heatmap(data_cm, annot = True, fmt = "d")
# plt.title("Confusion Matrix Random Forest")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()

# prediction
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X_cv = cv.fit_transform(data['lemmatize_review'])
y_cv = data['label']

X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_cv, y_cv, test_size=0.3, random_state=42)

clf = MultinomialNB()
clf.fit(X_train_cv, y_train_cv)
clf.score(X_test_cv, y_test_cv)

data = ["bagus banget", "suka banget", "tidak sesuai deskripsi","rugi beli disini","pengiriman cepat","jelek","buruk"]
vect = cv.transform(data).toarray()
new_pred = clf.predict(vect)
print(new_pred)


vect2 = tfidf.transform(data).toarray()
new_pred2 = clf.predict(vect2)
print(new_pred2)
