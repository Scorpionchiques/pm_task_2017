from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import pickle
target_names=['science','style', 'culture','life', 'economics', 'business', 'travel', 'forces', 'media', 'sport']





with open('data.pickle', 'rb') as handle:
    data = pickle.load(handle)
x_train=data[0]
y_train=data[1]


with open('test.pickle', 'rb') as handle:
    test = pickle.load(handle)

x_test=test




stopWords = stopwords.words('russian')

text_clf = Pipeline([('vect', CountVectorizer(stop_words=stopWords)), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

text_clf = text_clf.fit(x_train, y_train)

predicted = text_clf.predict(x_test)
with open('result1.txt', 'w+', encoding='utf-8') as file:
    for pred in predicted:
        file.write(pred + '\n')
