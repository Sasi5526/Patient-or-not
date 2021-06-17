# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 00:44:53 2021

@author: sasim
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

#Importing CSV files
train_dataset=pd.read_csv('D:\\sasi\\NLP Data Test\\NLP Data Test\\Train.csv')
test_dataset=pd.read_csv('D:\\sasi\\NLP Data Test\\NLP Data Test\\Test.csv')

x = train_dataset['Conversations']
y = train_dataset['Patient Or not']
x_test = test_dataset['Conversations']

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words=None,
                             ngram_range=(1, 1), min_df=2, max_df=0.4, binary=True)

x = vectorizer.fit_transform(x)


x_test = vectorizer.transform(x_test)

pickle.dump(vectorizer, open('tranform.pkl', 'wb'))



#Splitting the train into another train and validation sets in 70:30 ratio respectively

X_train, X_valid, y_train, y_valid = train_test_split(x, y, \
                                                    test_size=0.3, random_state=42)






#Applying Bernoulli's Naive Bayes
model = BernoulliNB(fit_prior=True)
model.fit(X_train, y_train)

valid_preds = model.predict(X_valid)
print(classification_report(y_valid, valid_preds))
print(f'Accuracy:{accuracy_score(y_valid, valid_preds)}')

filename = 'nlp_model.pkl'
pickle.dump(model, open(filename, 'wb'))


#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
#
clf = MultinomialNB()
clf.fit(X_train, y_train)
clf.score(X_valid,y_valid)
filename = 'nlp_model.pkl'
pickle.dump(clf, open(filename, 'wb'))
    