#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import warnings
warnings.filterwarnings('ignore')


# In[2]:


lc = pd.read_csv('loan_data_2007_2014.csv')
lc.head()


# In[3]:


lc.shape


# In[4]:


# check if there any duplicated columns
lc.duplicated().sum()


# In[5]:


# no need to remove duplicated columns


# In[6]:


# check the missing values in our dataset
missing = lc.isnull().sum()/lc.shape[0]*100
missing = missing.sort_values(ascending=False)
missing[missing>0]


# In[7]:


# remove columns whose missing values more than 40%
lc.dropna(thresh=lc.shape[0]*0.6, axis=1, inplace=True)
lc.shape


# In[8]:


# divide into categorical and numerical column for more data exploration
lc_cat = lc.select_dtypes(include=['object']).columns
lc_num = lc.select_dtypes(include=['float64','int64']).columns


# In[9]:


# explore categorical columns
lc[lc_cat].describe().T


# In[10]:


# remove columns whose too many unique values
lc.drop(['emp_title','url','title','zip_code'], axis=1, inplace=True)

# remove columns whose only 1 unique value
lc.drop('application_type', axis=1, inplace=True)

# remove columns whose particular value that is dominant
lc.drop('pymnt_plan', axis=1, inplace=True)

# remove columns which are considered useless as predictor
lc.drop(['issue_d','earliest_cr_line','last_pymnt_d','last_credit_pull_d'], axis=1, inplace=True)

lc.shape


# In[11]:


# explore numerical columns
lc[lc_num].describe().T


# In[12]:


# remove identity columns
lc.drop(['Unnamed: 0','id','member_id'], axis=1, inplace=True)
lc.shape


# In[13]:


# define target label
lc['loan_status'].value_counts(normalize=True)


# In[14]:


# assume charged off, Late (31-120 days), Late (16-30 days), Default, and-
# Does not meet the credit policy. Status:Charged Off as BAD BORROWERS
lc['loan_status'] = np.where(lc.loc[:,'loan_status'].isin
                               (['Charged Off','Default','Late (31-120 days)','Late (16-30 days)',
                                'Does not meet the credit policy. Status:Charged Off'
                                ]),1,0)

lc['loan_status'].value_counts(normalize=True)


# In[15]:


# okay, target label has been defined!


# In[16]:


lc['loan_status'].head()


# In[17]:


# split data into training and testing set
X = lc.drop('loan_status', axis=1)
y = lc['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)


# In[18]:


X_train.shape, X_test.shape


# In[19]:


y_train.value_counts(normalize=True)


# In[20]:


y_test.value_counts(normalize=True)


# In[21]:


# conduct some data cleaning
cat = X_train.select_dtypes(include=['object']).columns
num = X_train.select_dtypes(include=['float64','int64']).columns


# In[22]:


for col in cat:
    print(col, ' -----------> ', X_train[col].nunique())
    print(X_train[col].unique())
    print()


# In[23]:


# clean column which consist number and change it's data type
X_train['term'] = pd.to_numeric(X_train['term'].str.replace(' months', ''))

X_train['emp_length'] = X_train['emp_length'].str.replace(' years', '')
X_train['emp_length'] = X_train['emp_length'].str.replace(' year', '')
X_train['emp_length'] = X_train['emp_length'].str.replace('+', '')
X_train['emp_length'] = X_train['emp_length'].str.replace('< 1', '0')
X_train['emp_length'].fillna(value=0, inplace=True)
X_train['emp_length'] = pd.to_numeric(X_train['emp_length'])


# In[24]:


X_train[['term', 'emp_length']].info()


# In[25]:


X_train[cat].isnull().sum()


# In[26]:


# conduct the same data cleaning for testing set too


# In[27]:


# clean column which consist number and change it's data type
X_test['term'] = pd.to_numeric(X_test['term'].str.replace(' months', ''))

X_test['emp_length'] = X_test['emp_length'].str.replace(' years', '')
X_test['emp_length'] = X_test['emp_length'].str.replace(' year', '')
X_test['emp_length'] = X_test['emp_length'].str.replace('+', '')
X_test['emp_length'] = X_test['emp_length'].str.replace('< 1', '0')
X_test['emp_length'].fillna(value=0, inplace=True)
X_test['emp_length'] = pd.to_numeric(X_test['emp_length'])


# In[28]:


X_test[['term', 'emp_length']].info()


# In[29]:


X_test[cat].isnull().sum()


# In[30]:


# handle missing values in numerical column
num_miss = X_train[num].isnull().sum()[X_train[num].isnull().sum()>0]
num_miss


# In[31]:


X_train[num_miss.index] = X_train[num_miss.index].fillna(X_train[num_miss.index].median())


# In[32]:


num_miss = X_train[num].isnull().sum()[X_train[num].isnull().sum()>0]
num_miss


# In[33]:


num_miss = X_test[num].isnull().sum()[X_test[num].isnull().sum()>0]
num_miss


# In[34]:


X_test[num_miss.index] = X_test[num_miss.index].fillna(X_test[num_miss.index].median())


# In[35]:


num_miss = X_test[num].isnull().sum()[X_test[num].isnull().sum()>0]
num_miss


# In[36]:


cat = X_train.select_dtypes(include=['object']).columns
num = X_train.select_dtypes(include=['float64','int64']).columns


# In[37]:


# feature encoding
encoder = LabelEncoder()
for col in cat:
    encoder.fit(X_train[col].unique())
    X_train[col] = encoder.transform(X_train[col])
    X_test[col] = encoder.transform(X_test[col])


# In[38]:


# feature scaling
scaler = StandardScaler()
for col in num:
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)


# In[39]:


# decision tree classifier model
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
dtree_preds = dtree.predict(X_test)
print('model : DecisionTreeClassifier')
print('accuracy score : %', round(accuracy_score(y_test,dtree_preds)*100,2)) 
print('f1 score : %', round(f1_score(y_test,dtree_preds,average='micro')*100,2))
print('precision score : %', round(precision_score(y_test,dtree_preds,average='micro')*100,2))
print('recall score : %', round(recall_score(y_test,dtree_preds,average='micro')*100,2))
print()


# In[40]:


# feature importance of decision tree classifier model
dtree_fi = pd.DataFrame({'feature' : X.columns, 'importance' : dtree.feature_importances_})
dtree_fi = dtree_fi.sort_values('importance', ascending=True)
dtree_fi = dtree_fi.tail(10)
dtree_fi.plot(x='feature', y='importance', kind='barh', figsize=(10, 6))


# In[41]:


# logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg_preds = logreg.predict(X_test)
print('model : LinerRegression')
print('accuracy score : %', round(accuracy_score(y_test,logreg_preds)*100,2)) 
print('f1 score : %', round(f1_score(y_test,logreg_preds,average='micro')*100,2))
print('precision score : %', round(precision_score(y_test,logreg_preds,average='micro')*100,2))
print('recall score : %', round(recall_score(y_test,logreg_preds,average='micro')*100,2))
print()


# In[42]:


# feature importance of logistic regression model
logreg_fi = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(logreg.coef_[0])})
logreg_fi = logreg_fi.sort_values('Importance', ascending=True)
logreg_fi = logreg_fi.tail(10)
logreg_fi.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))

