# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:12:50 2017

@author: Gunnvant
"""

import os
import pandas as pd
import re
import numpy as np

base_dir='E:\Work\Python\Python Trainings\Python Advanced\Data'

os.chdir(base_dir)

## Cleaning the data
jokes=pd.read_csv('jokes_scrapped.csv')
p=re.compile(r'\d\.\d{2}')
re.findall(p,jokes.Rating[0])
jokes['Ratings_Cleaned']=jokes.Rating.map(lambda x: re.findall(p,x))
#jokes.ix[[387,388,389,390]]['Ratings_Cleaned'].apply(lambda x:len(x))
jokes['Ratings_Cleaned']=jokes['Ratings_Cleaned'].map(lambda x:np.nan if len(x)==0 else x[0])
jokes.to_csv('jokes_cleaned.csv',index=False)

## This will be the read operation for file
jokes_cleaned=pd.read_csv('jokes_cleaned.csv')
jokes_cleaned.isnull().sum()
jokes_cleaned=jokes_cleaned.dropna()

jokes_cleaned.dtypes

jokes_cleaned.Ratings_Cleaned.describe()

## Assume if the rating is 4 or more, the joke is funny
jokes_cleaned['Funny']=jokes_cleaned['Ratings_Cleaned'].map(lambda x: 1 if x>=4 else 0)
jokes_cleaned['Funny'].value_counts()/jokes_cleaned.shape[0]
## We will build a model to predict if a given joke is funny or not

## Data prepration
# Creating tfidf/count matrix 

from sklearn.feature_extraction.text import TfidfVectorizer

tf=TfidfVectorizer(stop_words='english')

text=['This is sentence 1. And may be is a very good sentence.','This is sentence 2. Do you think it is good or bad?','This might be sentence three. Life is very cruel but cats are cute!!']

matrix=tf.fit_transform(text)
tf.get_feature_names()

pd.DataFrame(data=matrix.toarray(),columns=tf.get_feature_names())

X=tf.fit_transform(jokes_cleaned['Joke'])
y=jokes_cleaned['Funny']
## Create test_train split

import sklearn.model_selection as model_selection

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.20,random_state=300)

## Let's create a logistic regression model and use regularisation

from sklearn.linear_model import LogisticRegression

clf=LogisticRegression(penalty='l2',C=2)

clf.fit(X_train,y_train)

clf.score(X_test,y_test)

## Let's do a grid search and see if we can improve our choice of C
np.random.seed(300)
clf1=model_selection.GridSearchCV(LogisticRegression(),param_grid={'penalty':['l2','l1'],'C':np.random.uniform(0.1,120,100)})

clf1.fit(X_train,y_train)

clf1.best_params_

clf1.score(X_test,y_test)

# metrics module
import sklearn.metrics as metrics
metrics.roc_auc_score(y_test,clf1.predict_proba(X_test)[:,1])
print metrics.classification_report(y_test,clf1.predict(X_test))

# We are doing just marginally better than a guessing strategy, cen we improve?

tf=TfidfVectorizer(stop_words='english',ngram_range=(1,2))

text=['This is sentence 1. And may be is a very good sentence.','This is sentence 2. Do you think it is good or bad?','This might be sentence three. Life is very cruel but cats are cute!!']

matrix=tf.fit_transform(text)
tf.get_feature_names()

pd.DataFrame(data=matrix.toarray(),columns=tf.get_feature_names())

X=tf.fit_transform(jokes_cleaned['Joke'])

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.20,random_state=300)

np.random.seed(300)
clf2=model_selection.GridSearchCV(LogisticRegression(),param_grid={'penalty':['l2','l1'],'C':np.random.uniform(0.1,120,50)},scoring='roc_auc')
clf2.fit(X_train,y_train)
clf2.best_params_
clf2.score(X_test,y_test)

metrics.accuracy_score(y_test,clf2.predict(X_test))
print metrics.classification_report(y_test,clf2.predict(X_test))
metrics.roc_auc_score(y_test,clf2.predict_proba(X_test)[:,1])

## Doesn't improve much
## Multiclass logistic 