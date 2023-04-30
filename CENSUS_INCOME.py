#!/usr/bin/env python
# coding: utf-8

# # IMPORT IMPORTANT LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# # IMPORTING DATASET

# In[2]:


df=pd.read_csv('CENSUS_INCOME.csv')
df.head()


# In[3]:


df.tail()


# # FEATURE OF DATASET

# Project Description
# 
# This data was extracted from the 1994 Census bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0)). 
# 
# The prediction task is to determine whether a person makes over $50K a year.
# 
# Description of fnlwgt (final weight)
# 
# The weights on the Current Population Survey (CPS) files are controlled to independent estimates of the civilian non-institutional population of the US. These are prepared monthly for us by Population Division here at the Census Bureau. We use 3 sets of controls. These are:
# 
# 1.A single cell estimate of the population 16+ for each state.
# 
# 2.Controls for Hispanic Origin by age and sex.
# 
# 3.Controls by Race, age and sex.
# 
# We use all three sets of controls in our weighting program and "rake" through them 6 times so that by the end we come back to all the controls we used. The term estimate refers to population totals derived from CPS by creating "weighted tallies" of any specified socio-economic characteristics of the population. People with similar demographic characteristics should have similar weights. There is one important caveat to remember about this statement. That is that since the CPS sample is actually a collection of 51 state samples, each with its own probability of selection, the statement only applies within state.

# # ROW AND COLUMNS

# In[4]:


print("NUMBER OF ROW : ",df.shape[0])
print("NUMBER OF COLUMNS : ",df.shape[1])


# # Exploratory Data Analysis and Data Processing

# # IDENTIFIED THE DATA TYPE

# In[5]:


df.info()


# AS WE SEEN THEIR IS NO MISSING VALUE IN THE DATASET

# In[6]:


plt.figure(figsize=(15,9))
sns.heatmap(df.isnull())


# In[7]:


df.nunique()


# In[8]:


df.duplicated().sum()


# In[9]:


df = df.drop_duplicates()


# In[10]:


df.duplicated().sum()


# In[11]:


for col in df.columns:
    if df[col].isna().any():
        print(f"{col} column contains missing values")


# # START DATA VISUALIZATION

# # 1.) UNIVARIATE ANALAYSIS

# In[12]:


for i in df.columns:
    if df[i].dtype=='O':
        plt.figure(figsize=(15,9))
        sns.countplot(df[i])
        plt.xticks(rotation=45,fontweight="bold")
        plt.yticks(fontweight="bold")
    if df[i].dtype=='int64':
        plt.figure(figsize=(15,9))
        sns.distplot(df[i])
        plt.xticks(fontweight="bold")
        plt.yticks(fontweight="bold")


# OBSERVATION:-
#     
#     * AS WE SEEN MOST OF THE PEOPLE ARE YOUNG AND BELONG TO (20-40 YEAR)
#     
#     * AS WE SEEN MOST OF THE PEOPLE BELONGS TO PRIVATE CLASS
#     
#     * AS WE SEEN MOST OF THE PEOPLE ARE HIGHLY GRADUATE THEN FOLLOWED BY SOME COLLEGE,THEN BECHLORE
#     
#     * AS WE SEEN MOST OF THE PEOPLE EDUCATION LEVEL IS IN BETWEEN 7.5 TO 10
#     
#     * AS WE SEEN MARITAL STATUS OF MOST OF THE PEOPLE IS MARRIED-CIV-SPOUSE THEN NEVER-MARRIED, DIVORCED
#     
#     * OCCUPATION OF MOST OF THE PEOPLE IS CRAFT-REPAIR,PROF-SPECIALITY,EXEC-MANAGERAL
#     
#     * IN DATASET MOST OF THE PEOPLE ARE HUSBAND , THEN NOT IN FAMILY CATEGORY
#     
#     * MOST OF THE PEOPLE ARE FROM WHITE RACE,THEN BLACK RACE
#     
#     * MOST OF THE PEOPLE ARE MALE 
#     
#     * CAPITAL GAIN OF MOST OF THE PEOPLE IS BETWEEN 0 TO 20000
#     
#     * MOST OF THE PEOPLE WORKING 40 HOURS PER WEEK
#     
#     * MOST OF THE PEOPLE ARE BELONGS TO UNITED STATE
#     
#     * IN DATASET MOST OF THE PEOPLE INCOME <=50

# # 2.) BIVARIATE ANALAYSIS

# In[18]:


pd.crosstab(df['Sex'],df['Income'],margins=False).plot(kind='bar',figsize=(15,9))
plt.yticks(fontweight="bold")
plt.xticks(fontweight='bold',rotation=0)
plt.title('Gender VS Income',fontweight='bold',fontsize=20)
plt.xlabel('SEX',fontsize=18,fontweight="bold")


# observation:-
#      
#         * AS WE SEEN IN THE ABOVE GRAPH MALE AND FEMALE BOTH HAVE <=50K SALARY HIGHER THEN >50K

# # BY MARITAL STATUS WITH INCOME

# In[29]:


pd.crosstab(df['Marital_status'],df['Income'],margins=False).plot(kind='bar',figsize=(15,9))
plt.yticks(fontweight="bold")
plt.xticks(fontweight='bold',rotation=45)
plt.title('Marital_status VS Income',fontweight='bold',fontsize=20)


# # BY WORKPLACE VS INCOME

# In[38]:


pd.crosstab(df['Workclass'],df['Income'],margins=False).plot(kind='bar',figsize=(15,9))
plt.yticks(fontweight="bold")
plt.xticks(fontweight='bold',rotation=45)
plt.title('WORKCLASS VS Income',fontweight='bold',fontsize=20)


# # BY RELATION VS INCOME

# In[40]:


pd.crosstab(df['Relationship'],df['Income'],margins=False).plot(kind='bar',figsize=(15,9))
plt.yticks(fontweight="bold")
plt.xticks(fontweight='bold',rotation=45)
plt.title('RELATIONSHIP VS Income',fontweight='bold',fontsize=20)


# # BY Occupation VS Income

# In[44]:


pd.crosstab(df['Occupation'],df['Income'],margins=False).plot(kind='bar',figsize=(15,9))
plt.yticks(fontweight="bold")
plt.xticks(fontweight='bold',rotation=45)
plt.title('Occupation VS Income',fontweight='bold',fontsize=20)


# In[45]:


pd.crosstab(df['Race'],df['Income'],margins=False).plot(kind='bar',figsize=(15,9))
plt.yticks(fontweight="bold")
plt.xticks(fontweight='bold',rotation=45)
plt.title('Race VS Income',fontweight='bold',fontsize=20)


# # 3.) multivariate

# In[47]:


sns.pairplot(df)


# In[50]:


df.drop(['Fnlwgt'],inplace=True,axis=1)


# # LETS FIND THE CORRELATION

# In[51]:


df.corr()


# # LET'S UNDERSTAND ON HEATMAP

# In[56]:


plt.figure(figsize=(15,9))
sns.heatmap(df.corr(),annot=True)


# # ENCODING OF DATAFRAME

# WE HAVE A CATEGORICAL DATA IN DATADRAME SO WE NEED TO ENCODE DATA INTO NUMERICAL DATA SO WILL BE UNDERSTABLE BY MACHINE

# In[57]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[58]:


for i in df.columns:
    if df[i].dtypes=="object":
        df[i]=le.fit_transform(df[i])
df.head()


# # REMOVING OUTLIERS

# In[59]:


plt.figure(figsize=(20,30),facecolor='white')
plotnumber=1
for i in df.columns:
    if plotnumber<=40:
        ax=plt.subplot(10,4,plotnumber)
        sns.boxplot(df[i],color='gold')
        plt.xlabel(i,fontsize=20)
    plotnumber+=1
plt.tight_layout()


# In[60]:


from scipy.stats import zscore
z=np.abs(zscore(df))
threshold=3
np.where(z>3)


# In[61]:


df_new=df[(z<3).all(axis=1)]
df_new


# In[62]:


df_new.shape


# In[63]:


df.shape


# In[65]:


data_loss=((32536-27702)/32536)*100
data_loss


# # 14% data is higher then the 10% so we can cont remove the outliers

# # seprating the columns into feature and target

# In[68]:


X=df.drop('Income',axis=1)
y=df['Income']


# # scaling the data using min max scaler

# In[67]:


from sklearn.preprocessing import MinMaxScaler
mms= MinMaxScaler()


# In[69]:


X=mms.fit_transform(X)


# In[70]:


X


# In[71]:


xf=pd.DataFrame(data=x)
xf


# # MACHINE LEARNING MODEL

# In[72]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# In[73]:


X_train.shape


# In[76]:


y_train.shape


# In[74]:


X_test.shape


# In[77]:


y_test.shape


# # with MultinomialNB

# In[78]:


mnb=MultinomialNB()
mnb.fit(X_train,y_train)
predmnb=mnb.predict(X_test)
predmnb


# In[79]:


print(accuracy_score(y_test,predmnb)*100)
print(confusion_matrix(y_test,predmnb))
print(classification_report(y_test,predmnb))


# # 2.) DECISION TREE CLASSIFIER

# In[80]:


from sklearn.tree import DecisionTreeClassifier


# In[81]:


dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
preddtc=dtc.predict(X_test)
preddtc


# In[82]:


print(accuracy_score(y_test,preddtc)*100)
print(confusion_matrix(y_test,preddtc))
print(classification_report(y_test,preddtc))


# # 3.) SVM(SUPPORT VECTOR MACHINE)

# In[83]:


from sklearn.svm import SVC


# In[84]:


sv=SVC()
sv.fit(X_train,y_train)
predsv=sv.predict(X_test)
predsv


# In[85]:


print(accuracy_score(y_test,predsv)*100)
print(confusion_matrix(y_test,predsv))
print(classification_report(y_test,predsv))


# # 4.)KNeighborsClassifier

# In[86]:


from sklearn.neighbors import KNeighborsClassifier


# In[87]:


knn= KNeighborsClassifier()
knn.fit(X_train,y_train)
predknn=knn.predict(X_test)
predknn


# In[88]:


print(accuracy_score(y_test,predknn)*100)
print(confusion_matrix(y_test,predknn))
print(classification_report(y_test,predknn))


# # 5.)logestic regression

# In[89]:


from sklearn.linear_model import LogisticRegression


# In[90]:


lr=LogisticRegression()
lr.fit(X_train,y_train)
predlr=lr.predict(X_test)


# In[91]:


print(accuracy_score(y_test,predlr)*100)
print(confusion_matrix(y_test,predlr))
print(classification_report(y_test,predlr))


# # AS WE SEEN SVM HAS HIGHEST SCORE

# # SAVING THE MODEL

# In[95]:


import joblib
joblib.dump(sv,"CENSUS_INCOME.pkl")


# # PREDICTION

# In[97]:


model=joblib.load("CENSUS_INCOME.pkl")

#perdiction
prediction=model.predict(X_test)
prediction


# In[99]:


pd.DataFrame([model.predict(X_test)[:],y_test[:]],index=["Predicted","Original"])


# # THANK YOU

# In[ ]:




