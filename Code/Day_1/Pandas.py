# -*- coding: utf-8 -*-
"""
Created on Wed May 04 14:30:03 2016

@author: Gunnvant
"""

#### Pandas datastructures
## Series
# Creating series
# Accessing data in series
# Adding data in a series
# Deleting data from a series
# Changing the dtypes in a series

l1=range(11)
print l1

#Add one to each element
l1+1

#Write a loop
l2=[]
for l in l1:
    l2.append(l+1)
print l2

#List comprehension
[x+1 for x in l1]

#Map
map(lambda x:x+1,l1)

#When we do data analysis we need our operations to be vectorised, obviously lists are not a good datastructure for data analysis. Series are like lists but permit vectorised operations.

import pandas as pd
from pandas import Series,DataFrame
import numpy as np

#Creating a Series object
#Genral syntax is:
# s= Series(data,indexindex)
# data:Dictionary, array, scalar


#Series from an array/list
S=Series([1,2,3,4,5])
S
print S[0]
print S[2]
S.values
S.index

S=Series([1,2,3,4,5],index=['a','b','c','d','e'])
S
S[0]
S['a']
S.values
S.index
#Adding one element to the series
S['g']=7
#Adding  more than one element to the series
#Use the append method to append a sub series to the bigger series
s=Series([8,9,10],index=['h','i','j'])
S=S.append(s)
S
#Deleting elements from a series
S=S.drop('g')
#More than one element from a series
S=S.drop(['i','j'])
S
#Series from a dictionary
d={'a':3,'b':4}
S1=Series(d)
S1
#We can make changes to a series in place also
S1['a']=4
S1

#Series support vectorization 
S1+1
S*2
S**2
np.log(S)
S+S1


#Series is built upon numpy arrays, one can efficiently change the dtype of the numpy array and then convert that array into a Series object
SC=Series(['1','2','3','4','5'])
SC
SC.dtype

SC=SC.astype('float64')
SC.dtype

##Corner case
s=Series(['1',2,3,4,np.nan,np.nan])
s=s.astype('int64')
s=s.astype('float64')

###DataFrames
# Creating data frames
# Accessing rows and columns of a data frame
# Changing name of a dataframe column
# Basic DataFrame methods
# Changing dtypes of a dataframe column

#When analysing data apart from the ability to work with vectorized data, the ability to work with the tabular data is also required. DataFrames is the tabular datastructure used to work with data.

# Creating a dataframe
# From a dictionary of lists, dictionaries, series
# From a list of dictionaries


d={'col1':[1,2,3,4,5],'col2':[6,7,8,9,10]}
frame1=DataFrame(d)

d1={'col1':{'r1':1,'r2':2},'col2':{'r1':3,'r2':4}}
frame2=DataFrame(d1)

d2={'col1':Series([1,2,3,4,5]),'col2':Series([6,7,8,9,10])}
frame3=DataFrame(d2)

l1=[{'col1':1,'col2':2,'col3':3},{'col1':4,'col2':5,'col3':6}]
frame4=DataFrame(l1)

# Accessing rows and columns of a data frame
frame3['col2']
type(frame3['col2'])
frame3[['col2']]
type(frame3[['col2']])
frame3[['col2','col1']]
type(frame3[['col2','col1']])

frame3.loc[1,'col1']
frame3.iloc[[1,0]]['col2']



# Changing names of columns in a DataFrame
frame1

#We can use rename() method to rename a column, rename accepts a dictionary 
frame1.rename(columns={'col1':'one','col2':'two'})
frame1

frame1.rename(columns={'col1':'one','col2':'two'},inplace=True)
frame1

frame1.columns=['Col1','Col2']
frame1

#One at a time/a few at a time
frame1.rename(columns={'Col1':'one'},inplace=True)
frame1
# Basic DataFrame methods
frame3.columns

frame3.head()

frame3.describe()

frame3.info()

frame3.dtypes


# Changing dtypes of a dataframe
# Each column in a dataframe is a series object, we can extract the series object 
frame3['col1']=Series(frame3['col1'].values.astype(float))

frame3.dtypes

# A less verbose implimentation
frame3['col2']=frame3['col2'].astype('float64')
frame3.dtypes

frame5=DataFrame({'col1':['1','2','3','4','5'],'col2':['6','7','8','9','10']})

frame5['col2']=frame5['col2'].astype(float)
frame5['col1']=frame5['col1'].astype(float)
frame5.dtypes

## Data Import
import os
os.getcwd()

#Setting directory to a particular path
os.chdir('F:\\Work\\Jigsaw Academy\\Data Scientist Course\\Data Science Redo\\Live Classes\\Intro to R')

## Flat files
# Checking the delimiter
# Checking the header
# Checking how missing values are populated

#We will use pd.read_table() method to do imports
dat1=pd.read_table('sample2.csv',sep=',',header=0)
dat1.head()

dat1.dtypes

dat2=pd.read_table('sample1.txt')
dat2.head()

dat2.dtypes

dat2=pd.read_table('sample1.txt',na_values=['Missing'])
dat2.head()

dat2.dtypes
