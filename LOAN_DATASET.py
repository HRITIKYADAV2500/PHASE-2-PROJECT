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


df=pd.read_csv('LOAN.csv')
df.head()


# In[3]:


df.tail()


# # FEATURE OF DATASET

# Project Description
# 
# This dataset includes details of applicants who have applied for loan. The dataset includes details like credit history, loan amount, their income, dependents etc. 
# 
# Independent Variables:
# 
# 1.Loan_ID - This refer to the unique identifier of the applicant's affirmed purchases
# 
# 2.Gender - This refers to either of the two main categories (male and female) into which applicants are divided on the basis of 
# their reproductive functions
# 
# 3.Married - This refers to applicant being in a state of matrimony
# 
# 4.Dependents - This refres to persons who depends on the applicants for survival
# 
# 5.Education - This refers to number of years in which applicant received systematic instruction, especially at a school or university
# 
# 6.Self_Employed - This refers to applicant working for oneself as a freelancer or the owner of a business rather than for an employer
# 
# 7.Applicant Income - This refers to disposable income available for the applicant's use under State law.
# 
# 8.CoapplicantIncome - This refers to disposable income available for the people that participate in the loan application process 
# alongside the main applicant use under State law.
# 
# 9.Loan_Amount - This refers to the amount of money an applicant owe at any given time.
# 
# 10.Loan_Amount_Term - This refers to the duaration in which the loan is availed to the applicant
# 
# 11.Credit History - This refers to a record of applicant's ability to repay debts and demonstrated responsibility in repaying them.
# 
# 12.Property_Area - This refers to the total area within the boundaries of the property as set out in Schedule.
# 
# 13.Loan_Status - This refres to whether applicant is eligible to be availed the Loan requested.
# You have to build a model that can predict whether the loan of the applicant will be approved(Loan_status) or not on the basis of the details provided in the dataset.

# # ROW AND COLUMNS

# In[4]:


print("NUMBER OF ROW : ",df.shape[0])
print("NUMBER OF COLUMNS : ",df.shape[1])


# # Exploratory Data Analysis and Data Processing

# # IDENTIFIED THE DATA TYPE

# In[5]:


df.info()


# # CHECKING MISSING VALUE IN COLUMNS

# In[6]:


for col in df.columns:
    if df[col].isna().any():
        print(f"{col} column contains missing values")


# # FILL MISSING VALUE IN COLUMNS BY FILLNA METHOD

# In[7]:


df['Gender'].fillna(value=df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(value=df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(value=df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(value=df['Self_Employed'].mode()[0],inplace=True)
df['LoanAmount'].fillna(value=df['LoanAmount'].median(),inplace=True)
df['Loan_Amount_Term'].fillna(value=df['Loan_Amount_Term'].median(),inplace=True)
df['Credit_History'].fillna(value=df['Credit_History'].median(),inplace=True)


# In[8]:


print(df.isna().sum())


# NOW THERE IS NO NULL VALUE IN DATASET

# In[9]:


plt.figure(figsize=(15,9))
sns.heatmap(df.isnull())


# In[10]:


df.nunique()


# In[11]:


df.duplicated().sum()


# # THERE IS NO NEED OF LOAN ID COLUMN SO WE HAVE TO DROP THAT

# In[12]:


df.drop('Loan_ID',axis=1,inplace=True)


# # START DATA VISUALIZATION

# # 1.) UNIVARIATE ANALAYSIS

# In[13]:


for i in df.columns:
    if df[i].dtype=='O':
        plt.figure(figsize=(15,9))
        sns.countplot(df[i])
        plt.xticks(fontweight="bold")
        plt.yticks(fontweight="bold")
    if df[i].dtype=='int64':
        plt.figure(figsize=(15,9))
        sns.distplot(df[i])
        plt.xticks(fontweight="bold")
        plt.yticks(fontweight="bold")


# OBSERVATION:-
#     
#     * AS WE SEEN IN THE GRAPH MALE HAS HIGHER NUMBER THEN WOMEN
#     
#     * AS WE SEEN MARIED HAS HIGHER VALUE THEN UNMARIED
#     
#     * AS WE SEEN MOST PEOPLE DEPENDENTS ON ZERO PERSON
#     
#     * AS WE SEEN AS WE SEEN MOST  OF THE PEOPLE ARE GRADUATED
#     
#     * AS WE SEEN MOST OF THE PEOPLE ARE NOT SELF EMPLOYED
#     
#     * AS WE SEEN mmost of the people has income between 0 to 20000
#     
#     * AS WE SEEN MOST OF THE PEOPLE BELONGS TO SEMIURBAN THEN URBAN
#     
#     * AS WE SEEN MOST OF THE PEOPLE LOAN HAS BEEN PASSED

# # 2.) BIVARIATE ANALAYSIS

# # BY GENDER vS LOAN STATUS

# In[22]:


pd.crosstab(df['Gender'],df['Loan_Status'],margins=False).plot(kind='bar',figsize=(15,8))
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.title('Gender Vs Loan Status',fontsize=20,fontweight='bold')


# # OBSERVATION:-
#     
#     * AS WE SEEN MALE HAS HIGH LOAN STATUS THEN FEMALE
#     * IN BOTH THE CASE THEIR IS LOW CHANCE OF CANCLE THE LOAN

# # BY MARRIED VS LOAN STATUS

# In[24]:


pd.crosstab(df['Married'],df['Loan_Status'],margins=False).plot(kind='bar',figsize=(15,8))
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.title('Married Vs Loan Status',fontsize=20,fontweight='bold')


#  * AS WE SEEN MARRIED HAS HIGH CHANCE TO GET A LOAN 

# # BY DEPENDETS

# In[26]:


pd.crosstab(df['Dependents'],df['Loan_Status'],margins=False).plot(kind='bar',figsize=(15,8))
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.title('Dependets Vs Loan Status',fontsize=20,fontweight='bold')


# # BY EDUCATION

# In[28]:


pd.crosstab(df['Education'],df['Loan_Status'],margins=False).plot(kind='bar',figsize=(15,8))
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.title('Education Vs Loan Status',fontsize=20,fontweight='bold')


# AS GRADUATE HAS THE HIGHER CHANCE WHO GET A LOAN EASILY

# # BY SELF EMPLOCYED

# In[30]:


pd.crosstab(df['Self_Employed'],df['Loan_Status'],margins=False).plot(kind='bar',figsize=(15,8))
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.title('Self_Employed Vs Loan Status',fontsize=20,fontweight='bold')


# # MULTIVARIATE ANALAYSIS

# In[31]:


sns.pairplot(df)


# # ENCODING OF DATAFRAME

# WE HAVE LOT OF CATEGORICAL DATA IN DATADRAME SO WE NEED TO ENCODE DATA INTO NUMERICAL DATA SO WILL BE UNDERSTABLE BY MACHINE

# In[40]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[41]:


for i in df.columns:
    if df[i].dtypes=="object":
        df[i]=le.fit_transform(df[i])
df.head()


# # seprating the columns into feature and target

# In[42]:


X=df.drop('Loan_Status',axis=1)


# In[43]:


y=df["Loan_Status"]


# # SCALING BY MIN MAX SCALING

# In[44]:


from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()


# In[45]:


X=mms.fit_transform(X)


# # MACHIME LEARNING MODEL

# In[46]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# In[47]:


X_train.shape


# In[48]:


y_train.shape


# In[49]:


X_test.shape


# In[50]:


y_test.shape


# # with MultinomialNB

# In[51]:


mnb=MultinomialNB()
mnb.fit(X_train,y_train)
predmnb=mnb.predict(X_test)
predmnb


# In[52]:


print(accuracy_score(y_test,predmnb)*100)
print(confusion_matrix(y_test,predmnb))
print(classification_report(y_test,predmnb))


# # 2.) DECISION TREE CLASSIFIER

# In[53]:


from sklearn.tree import DecisionTreeClassifier


# In[54]:


dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
preddtc=dtc.predict(X_test)
preddtc


# In[55]:


print(accuracy_score(y_test,preddtc)*100)
print(confusion_matrix(y_test,preddtc))
print(classification_report(y_test,preddtc))


# # 3.) SVM(SUPPORT VECTOR MACHINE)

# In[56]:


from sklearn.svm import SVC


# In[57]:


sv=SVC()
sv.fit(X_train,y_train)
predsv=sv.predict(X_test)
predsv


# In[58]:


print(accuracy_score(y_test,predsv)*100)
print(confusion_matrix(y_test,predsv))
print(classification_report(y_test,predsv))


# # 4.)KNeighborsClassifier

# In[59]:


from sklearn.neighbors import KNeighborsClassifier


# In[60]:


knn= KNeighborsClassifier()
knn.fit(X_train,y_train)
predknn=knn.predict(X_test)
predknn


# In[61]:


print(accuracy_score(y_test,predknn)*100)
print(confusion_matrix(y_test,predknn))
print(classification_report(y_test,predknn))


# # 5.)logestic regression

# In[62]:


from sklearn.linear_model import LogisticRegression


# In[63]:


lr=LogisticRegression()
lr.fit(X_train,y_train)
predlr=lr.predict(X_test)


# In[64]:


print(accuracy_score(y_test,predlr)*100)
print(confusion_matrix(y_test,predlr))
print(classification_report(y_test,predlr))


# # RESULT :-
#     
#     * LOGESTIC REGRESSION:-80%
#     * KNEIGHBOURS CLASSIFIER:-78%
#     * SVM(SUPPORT VECTOR MACHINE):-80%
#     * DECISION TREE CLASSIFIER:-69%
#     * with MultinomialNB:- 71%

# # SAVING THE MODEL

# In[66]:


import joblib
joblib.dump(lr,"LOAN_DATASET.pkl")


# # PREDICTION

# In[69]:


model = joblib.load("LOAN_DATASET.pkl")

# Prediction
prediction = model.predict(X_test)
prediction


# In[70]:


pd.DataFrame([model.predict(X_test)[:],y_test[:]],index=["Predicted","Original"])


# # END PROJECT 

# # THANK YOU

# In[ ]:





# In[ ]:




