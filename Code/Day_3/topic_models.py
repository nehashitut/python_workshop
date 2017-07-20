# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 22:25:00 2017

@author: Gunnvant
"""

import os
import numpy as np
import pandas as pd

base_dir='E:\Work\Python\Python Trainings\Python Advanced\Code\Day_3'

os.chdir(base_dir)

text=['China is a global hub for manufacturing, and is the largest manufacturing economy in the world as well as the largest exporter of goods in the world. China is also the worlds fastest growing consumer market and second largest importer of goods in the world',

"In 2011, China produced 1 tons of electronic waste. The annual amount is expected to increase by 1% as the Chinese economy grows. In addition to domestic waste production, large amounts of electronic waste are imported from overseas. Legislation banning the importation of electronic waste and requiring proper disposal of domestic waste has recently been introduced, but has been criticized as insufficient and susceptible to fraud. There have been local successes, such as in the city of Tianjin where 0.38 tons of electronic waste were disposed of properly in 2010, but much electronic waste is still improperly handled."]

from sklearn.feature_extraction.text import TfidfVectorizer
import re
pattern=re.compile(r'[0-9]')
text_cleaned=map(lambda x:re.sub(pattern,"",x),text)
tf=TfidfVectorizer(stop_words='english')
matrix=tf.fit_transform(text_cleaned)
pd.DataFrame(data=matrix.toarray(),columns=tf.get_feature_names())

from sklearn.decomposition import NMF

mod=NMF(n_components=2)

mod.fit(matrix)
mod.components_
mod.components_.shape
mod.transform(matrix)

topics=pd.DataFrame({'topic1':mod.components_[0],'topic2':mod.components_[1]},index=tf.get_feature_names())

mod.transform(matrix)

topics['topic1'].sort_values(ascending=False).head()

topics['topic2'].sort_values(ascending=False).head()

## Using lda

from sklearn.decomposition import LatentDirichletAllocation as lda

mod1=lda(n_topics=2)

mod1.fit(matrix)

mod1.components_

topics_lda=pd.DataFrame({'topic1':mod1.components_[0],'topic2':mod1.components_[1]},index=tf.get_feature_names())

mod1.transform(matrix)

topics_lda['topic1'].sort_values(ascending=False).head()

topics_lda['topic2'].sort_values(ascending=False).head()

## Visualising lda model

import pyLDAvis
import pyLDAvis.sklearn

p=pyLDAvis.sklearn.prepare(mod1, matrix, tf)

pyLDAvis.show(p)