# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 12:40:25 2017

@author: Gunnvant
"""

import os
import pandas as pd
os.chdir('E:\Work\Python\Python Trainings\Python Advanced\Data')

f=open('starbucks-menu-nutrition-food.csv','r')
dat=f.read()
f.close()
dat
dat.decode('utf-16',errors='ignore')
dat.decode('utf-16',errors='ignore').split("\n")

for d in dat.decode('utf-16',errors='ignore').split("\n")[0:-2]:
    print ",".join(d.split(','))

f=open('starbucks_cleaned.csv','w')
for d in dat.decode('utf-16',errors='ignore').split("\n")[0:-2]:
    f.write( ",".join(d.split(',')).encode('utf-8')+'\n')
f.close()

data=pd.read_csv('starbucks_cleaned.csv')
cols=data.columns
data.columns=map(lambda x:x.strip(),cols.tolist())

import re
pattern=re.compile(r'\(g\)|\.')
def sub(x):
    return re.sub(pattern,'',x)
data.columns=map(lambda x:x.strip(),map(sub,data.columns.tolist()))

data.to_csv('starbucks_final.csv',index=False)