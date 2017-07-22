# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 16:27:28 2017

@author: Gunnvant
"""

import os
import pandas as pd
import numpy as np
base_dir='E:\Work\Python\Python Trainings\Python Advanced\Data'
os.chdir(base_dir)
data=pd.read_json('jokes.json',encoding='utf-8')

## Count matrix and tfidf

doc=['This is document one','This is doucment two','This is document three']

import sklearn.feature_extraction.text as text

cv=text.CountVectorizer(doc)

cv.fit(doc)

cv.get_feature_names()

cv.transform(doc)

cv.transform(doc).toarray()

pd.DataFrame(data=cv.transform(doc).toarray(),columns=cv.get_feature_names())

tfid=text.TfidfVectorizer(doc)

tfid.fit(doc)

tfid.transform(doc).toarray()

pd.DataFrame(data=tfid.transform(doc).toarray(),columns=tfid.get_feature_names())

data.columns

y=data['Rank']
x=data['Raw_joke']

tfid=text.TfidfVectorizer(x.tolist())
tfid.fit(x.tolist())
print tfid.get_feature_names()
X=tfid.transform(x.tolist())

X.shape
y.head()

y=y.map(lambda x:1 if x>7 else 0)

import sklearn.linear_model as linear_model
clf=linear_model.LogisticRegression()
clf.fit(X,y)
clf.score(X,y)

import sklearn.model_selection as model_selection

clf1=model_selection.GridSearchCV(linear_model.LogisticRegression(),param_grid={'penalty':['l2','l1'],'C':np.random.uniform(0.1,120,100)})

clf1.fit(X,y)
clf1.best_params_
clf1.score(X,y)