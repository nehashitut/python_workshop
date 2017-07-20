# -*- coding: utf-8 -*-
"""
Created on Wed May 04 12:45:21 2016

@author: Gunnvant
"""

import pandas as pd
import numpy as np

import os
os.chdir('E:\Work\Python\Python Trainings')
#Download the data from following link
#https://s3.amazonaws.com/datasetsgun/data/terror.csv
data=pd.read_csv('terror.csv')

print data.head()
print data.shape
print data.columns.tolist()
## Data Manipulation tasks:
# Filtering data
# Selecting columns 
# Sorting data
# Adding new columns
# Group By aggregations
# Handling dates
# Handling text
# Merging dataframes
# Treating Missing Values


#### Filtering data
# Using logical subsets
# Using query to subset

### How many attacks happened in India
data['country_txt'].unique().tolist()
data[data['country_txt']=="India"]
data[data['country_txt']=="India"].shape[0]

## How many attacks happened in India and upto 3 people were killed
data[(data['country_txt']=='India')&(data['nkill']<=3)].shape[0]
# This looks ugly
data.query("country_txt=='India' & nkill<=3").shape[0]

# Extract the city and summary for attacks above
data.query("country_txt=='India' & nkill<=3")[['city','summary']].head(10)

# In a single terror incident in India, find out top 5 cities by number killed
data.query("country_txt=='India'").sort_values('nkill',ascending=False)['city'].head(5)
# Little more detailed
data.query("country_txt=='India'").sort_values('nkill',ascending=False)[['city','nkill','iyear']].head(5)

# In a single terror incident in India, find out top 5 cities by number killed and wounded
data.query("country_txt=='India'").sort_values(['nkill','nwound'],ascending=[False,False])['city'].head(5)


## Read the data called starbucks_final.csv
# An ideal diet should contain optimum level of nutrients
# can you find out the names of the items on menu that contain:
    #Upto 450 calories
    #Upto 40 g protein
    #Upto 10 g fat
    #Upto 40 g Carbs
    #Upto 5 g fibre
# Give the names of items on menu that satisfy the above criteria but contain least carbs but maximum protien

#### Adding new columns to the data
#Attacks which were successful and were suicide attacks
data['success_suicide']=data['suicide']+data['success']
data=data.assign(success_suicide_1=data['suicide']+data['success'])

data.query("success_suicide==2").shape[0]

### Dropping columns
data.drop('success_suicide_1',axis=1) 
data.drop('success_suicide_1',axis=1,inplace=True) 

### Aggregations and manipulations using apply and map
#map: map a function to each element of a series object
data['nkill'].describe()
#Suppose we want to label all the incidents where the number killed was more than 5 as severe. This would involve applying a function on each element of the series, map helps in doing that
def get_label(x):
    if x>5:
        return 'Severe'
    else:
        return 'Not Severe'
data['nkill'].map(get_label)
#You can use lambda functions as well (if else follows as special form when used with lambdas )
data['nkill'].map(lambda x: "Severe" if x>5 else "Not Severe")
# We can use apply to use a function row wise or column wise
# Let's write a function to label an incident that was both successful and suicidal

def get_label(row):
    if row['success']==1 and row['suicide']==1:
        return 1
    else:
        return 0
data.apply(get_label,axis=1)
data.apply(get_label,axis=1).unique()

data.apply(lambda row: 1 if row['success']==1 and row['suicide']==1 else 0,axis=1)

### Create a new  category representing if the incident occured in Afghanistan, Pakistan or India as one level of the category and all the other countries as another level

def get_label(row):
    if row['country_txt']=='India' or row['country_txt']=='Afghnistan' or row['country_txt']=='Pakistan':
        return 'Af-Pak-India'
    else:
        return 'ROW'
data.apply(get_label,axis=1)

data['Local']=data.apply(get_label,axis=1)
#### Group by aggregations
# Grouping by one or more variable(s) and aggregating one column
# Grouping by one or more variable(s) and aggregating multiple columns in same way
# Grouping by one or more variable(s) and aggregating multiple columns differently
#Finding average price across brands

## Number of incidents in Af-Pak-India vs ROW
data['Local'].value_counts()
data.groupby('Local').size()

## Number of suicide attacks by Af-Pak-India vs ROW
data.groupby(['Local','suicide']).agg({'eventid':np.size})
data.groupby(['Local','suicide'],as_index=False).agg({'eventid':np.size})

## Number of suicides attacks and average kills by Af-Pak-India vs ROW
data.groupby(['Local','suicide'])['nkill'].agg([np.mean,np.size])
data.groupby(['Local','suicide'])['nkill'].agg([np.mean,np.size]).reset_index()

#Rename columns
data.groupby(['Local','suicide'])['nkill'].agg([np.mean,np.size]).reset_index().rename(columns={'mean':'Average_Kills','size':'Number_Incidents'})

data.groupby(["Local",'suicide']).agg({'nkill':np.mean,'nwound':np.sum}).reset_index().rename(columns={'nwound':'Total_Wounded','nkill':'Number_Killed'})
## Use the file Python Data Manipulation_1.docx and attempt question1

#String manipulations
st=pd.read_csv("F:\\Work\\Jigsaw Academy\\Data Scientist Course\\Data Science Redo\\Live Classes\\Data Manipulation Visualisation\\Strings.csv")

print st.head()

st['Income_M'].mean()

st['Income_M']=st['Income_M'].str.replace("Rs","")
print st.head()

st['Income_M']=st['Income_M'].str.replace("/-","")
print st.head()

st['Income_M'].mean()

st.Income_M=pd.to_numeric(st.Income_M)
st.Income_M.mean()

#### Handling dates
os.chdir('E:\Work\Python\Python Trainings\Python Advanced\Data')
data=pd.read_csv('assignment_submission.csv')

data.columns

data['Started At'].head()
data['Finished At'].head()

## Time taken to complete a quiz
data['Started At']=data['Started At'].str.replace('UTC',"")
data['Finished At']=data['Finished At'].str.replace('UTC',"")

data['Started At']=pd.to_datetime(data['Started At'])
data['Finished At']=pd.to_datetime(data['Finished At'])

data['time_taken']=data['Finished At']-data['Started At']

data['time_taken'].head()

data['time_taken'].describe()

data['time_taken'].quantile(np.arange(0,0.99,0.01))

data['Started At'].dt.weekday #(Monday=0,...Sunday=6)
#Generic Time classes in pandas
pd.to_datetime('15-06-16')
pd.Timestamp('15-06-16')

# Time stamps are different from time intervals
pd.to_datetime('15-06-16')-pd.to_datetime('14-06-16')

a=pd.to_datetime('15-06-16')-pd.to_datetime('14-06-16')

a/365

a/pd.to_timedelta(365,unit='D')
#If time interval is added to a timestamp we will get a future timestamp

pd.Timestamp('15-06-16')+pd.to_timedelta(365,unit='D')


## Handling missing values
# Counting the number of missing values in each column
dat_m=pd.read_csv('F:\\Work\\Jigsaw Academy\\Credit Loan\\Credit.csv',na_values=['Missing',""])
# Number of missing values
dat_m.isnull().sum()

#Subsetting by missing values
dat_m[dat_m['MonthlyIncome'].isnull()]['DebtRatio']

#Replacing missing values
dat_m['age']=dat_m['age'].fillna(20)

## Joining data frames

dat1=data[['store','brand']]
dat2=data[['week','feat']]

pd.concat([dat1,dat2],axis=1)

dat3=dat1.ix[0:150]
dat4=dat1.ix[151:300]

pd.concat([dat3,dat4],axis=0)
pd.concat([dat3,dat4],axis=1)

## Merging DataFrames
df1=DataFrame({'CustomerID':[1,2,3,4,5,6],'Product':['Toaster','Toaster','Toaster','Radio','Radio','Radio']})
df2=DataFrame({'CustomerID':[2,4,6],'State':['Alabama','Alabama','Ohio']})

pd.merge(df1,df2,how='outer',on='CustomerID')

pd.merge(df1,df2,how='inner',on='CustomerID')

pd.merge(df1,df2,how='left',on='CustomerID')

pd.merge(df1,df2,how='right',on='CustomerID')

df1=DataFrame({'CustomerId':[1,2,3,4,5,6],'Product':['Toaster','Toaster','Toaster','Radio','Radio','Radio']})
df2=DataFrame({'CustomerID':[2,4,6],'State':['Alabama','Alabama','Ohio']})

pd.merge(df1,df2,how='inner',left_on='CustomerId',right_on='CustomerID').drop('CustomerID',axis=1)


