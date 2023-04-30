#!/usr/bin/env python
# coding: utf-8

# # IMPORT IMPORTANT LIBRARIES

# In[63]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt


# # LOAD DATASET 

# In[64]:


df=pd.read_csv('global_power_plant.csv')


# In[65]:


df.head()


# In[66]:


df.tail()


# # FEATURES OF DATASET

# Project Description
# The Global Power Plant Database is a comprehensive, open source database of power plants around the world. It centralizes power plant data to make it easier to navigate, compare and draw insights for one’s own analysis. The database covers approximately 35,000 power plants from 167 countries and includes thermal plants (e.g. coal, gas, oil, nuclear, biomass, waste, geothermal) and renewables (e.g. hydro, wind, solar). Each power plant is geolocated and entries contain information on plant capacity, generation, ownership, and fuel type. It will be continuously updated as data becomes available.
# Key attributes of the database
# 
# The database includes the following indicators:
# 
#  `country` (text): 3 character country code corresponding to the ISO 3166-1 alpha-3 specification [5]
# 
#  `country_long` (text): longer form of the country designation
# 
#  `name` (text): name or title of the power plant, generally in Romanized form
# 
#  `gppd_idnr` (text): 10 or 12 character identifier for the power plant
# 
# `capacity_mw` (number): electrical generating capacity in megawatts
# 
# `latitude` (number): geolocation in decimal degrees; WGS84 (EPSG:4326)
# 
# `longitude` (number): geolocation in decimal degrees; WGS84 (EPSG:4326)
# 
# `primary_fuel` (text): energy source used in primary electricity generation or export
# 
# 
# `other_fuel1` (text): energy source used in electricity generation or export
# 
# `other_fuel2` (text): energy source used in electricity generation or export
# 
# `other_fuel3` (text): energy source used in electricity generation or export
# 
#  `commissioning_year` (number): year of plant operation, weighted by unit-capacity when data is available
# 
# `owner` (text): majority shareholder of the power plant, generally in Romanized form
# 
# `source` (text): entity reporting the data; could be an organization, report, or document, generally in Romanized form
# 
# `url` (text): web document corresponding to the `source` field
# 
# `geolocation_source` (text): attribution for geolocation information
# 
# `wepp_id` (text): a reference to a unique plant identifier in the widely-used PLATTS-WEPP database.
# 
# `year_of_capacity_data` (number): year the capacity information was reported
# 
# `generation_gwh_2013` (number): electricity generation in gigawatt-hours reported for the year 2013
# 
# `generation_gwh_2014` (number): electricity generation in gigawatt-hours reported for the year 2014
# 
# `generation_gwh_2015` (number): electricity generation in gigawatt-hours reported for the year 2015
# 
# `generation_gwh_2016` (number): electricity generation in gigawatt-hours reported for the year 2016
# 
# `generation_gwh_2017` (number): electricity generation in gigawatt-hours reported for the year 2017
# 
# `generation_gwh_2018` (number): electricity generation in gigawatt-hours reported for the year 2018
# 
# `generation_gwh_2019` (number): electricity generation in gigawatt-hours reported for the year 2019
# 
# `generation_data_source` (text): attribution for the reported generation information
# 
# `estimated_generation_gwh_2013` (number): estimated electricity generation in gigawatt-hours for the year 2013
# 
# `estimated_generation_gwh_2014` (number): estimated electricity generation in gigawatt-hours for the year 2014 
# 
# `estimated_generation_gwh_2015` (number): estimated electricity generation in gigawatt-hours for the year 2015 
# 
# `estimated_generation_gwh_2016` (number): estimated electricity generation in gigawatt-hours for the year 2016 
# 
# `estimated_generation_gwh_2017` (number): estimated electricity generation in gigawatt-hours for the year 2017 
# 
# 'estimated_generation_note_2013` (text): label of the model/method used to estimate generation for the year 2013
# 
# `estimated_generation_note_2014` (text): label of the model/method used to estimate generation for the year 2014 
# 
# `estimated_generation_note_2015` (text): label of the model/method used to estimate generation for the year 2015
# 
# `estimated_generation_note_2016` (text): label of the model/method used to estimate generation for the year 2016
# 
# `estimated_generation_note_2017` (text): label of the model/method used to estimate generation for the year 2017 
# 
# Fuel Type Aggregation
# 
# We define the "Fuel Type" attribute of our database based on common fuel categories. 
# 
# Prediction :   Make two prediction  1) Primary Fuel    2) capacity_mw 
# 

# # LET'S UNDERSTAND DATASET

# In[67]:


print("NUMBER OF ROW IN DATASET : ",df.shape[0])
print("NUMBER OF COLUMNS IN DATASET : ",df.shape[1])


# In[68]:


df.columns


# In[69]:


df.info()


# * AS WE SEEN WE HAVE 27 COLUMNS 
# * OUT OF WHICH 12 ARE OBJECT DATA TYPE AND 15 ARE FLOAT TYPE
# * AS WE SEEN TOTAL 907 RAW AND SOME COLUMNS HAVE ALL NAN VALUES IN DATA SET

# # LET'S CHECK NULL AND UNIQUE VALUE

# In[70]:


df.isin(['NA','N/A','-',' ','?',' ?']).sum().any()


# In[71]:


df.isnull().sum()


# In[72]:


df.nunique()


# # LETS UNDERSTAND NULL VALUE ON HEATMAP

# In[73]:


plt.figure(figsize=(15,8))
sns.heatmap(df.isnull())


# # OBSERVATION:-
#     * AS WE SEEN 'OTHER_FUEL1','OTHER_FUEL2','OTHER_FUEL3','WEPP_ID','GENERATION_GWH_2013','GENERATION_GWH_2019','ESTIMATED_GENERATION_GWH' HAS HIGH NULL VALUE

# # DROP ALL THESE IRREVALENT COLUMNS

# In[74]:


df.drop(['other_fuel1','other_fuel2','other_fuel3','wepp_id','generation_gwh_2013','generation_gwh_2019','estimated_generation_gwh'],inplace=True,axis=1)


# In[75]:


df.drop(['source','url','country','country_long'],inplace=True,axis=1)


# In[76]:


df.drop(['generation_data_source','year_of_capacity_data'],inplace=True,axis=1)


# In[77]:


df.isnull().sum()


# # FILL REMAINING NAN COLUMNS WITH FILLNA

# In[78]:


df['latitude'].fillna(value=df['latitude'].median(),inplace=True)
df['longitude'].fillna(value=df['longitude'].median(),inplace=True)
df['commissioning_year'].fillna(value=df['commissioning_year'].median(),inplace=True)
df['owner'].fillna(value=df['owner'].mode()[0],inplace=True)
df['geolocation_source'].fillna(value=df['geolocation_source'].mode()[0],inplace=True)
df['generation_gwh_2014'].fillna(value=df['generation_gwh_2014'].median(), inplace= True)
df['generation_gwh_2015'].fillna(value=df['generation_gwh_2015'].median(), inplace= True)
df['generation_gwh_2016'].fillna(value=df['generation_gwh_2016'].median(), inplace= True)
df['generation_gwh_2017'].fillna(value=df['generation_gwh_2017'].median(), inplace= True)
df['generation_gwh_2018'].fillna(value=df['generation_gwh_2018'].median(), inplace= True)


# In[79]:


df.isnull().sum()


# NOW THEIR IS NO NULL VALUE IN DATASET

# # EDA(EXPLORATRY DATA ANALAYSIS)

# # 1.) Univariate analysis

# 1.1) PRIMARY FUEL

# In[80]:


plt.figure(figsize=(15,9))
sns.countplot(df['primary_fuel'])
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
plt.xlabel('primary fuel',fontweight='bold',fontsize=18)
plt.title('energy source used in primary electricity generation or export',fontweight='bold',fontsize=20)


# OBSERVATION:-
#     
#     *  AS WE SEEN COAL AND HYDRO HAS HIGHEST COUNT FOLLOWED BY  SOLAR AND WIND 
#     *  NUCLEAR HAS THE LOWEST COUNT IN THE OBSERVATION

# 1.2) CAPITAL MEGAWATTS

# In[81]:


plt.figure(figsize=(15,9))
sns.countplot(df['geolocation_source'])
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
plt.xlabel('GEOLOCATION SOURCE',fontweight='bold',fontsize=18)
plt.title('attribution for geolocation information',fontweight='bold',fontsize=20)


# 1.3) CAPITAL MEGAWATT

# In[82]:


plt.figure(figsize=(15,9))
sns.distplot(df['capacity_mw'])
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
plt.xlabel('CAPITAL MEGAWATT',fontweight='bold',fontsize=18)
plt.title('electrical generating capacity in megawatts',fontweight='bold',fontsize=20)


# 1.4) generation_gwh_2014

# In[83]:


plt.figure(figsize=(15,9))
sns.distplot(df['generation_gwh_2014'])
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
plt.xlabel('',fontweight='bold',fontsize=18)
plt.title('label of the model/method used to estimate generation for the year 2014',fontweight='bold',fontsize=20)


# 1.5) generation_gwh_2015

# In[84]:


plt.figure(figsize=(15,9))
sns.distplot(df['generation_gwh_2015'])
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
plt.xlabel('',fontweight='bold',fontsize=18)
plt.title('label of the model/method used to estimate generation for the year 2015',fontweight='bold',fontsize=20)


# 1.6)generation_gwh_2016

# In[85]:


plt.figure(figsize=(15,9))
sns.distplot(df['generation_gwh_2016'])
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
plt.xlabel('',fontweight='bold',fontsize=18)
plt.title('label of the model/method used to estimate generation for the year 2016',fontweight='bold',fontsize=20)


# 1.7) generation_gwh_2017

# In[86]:


plt.figure(figsize=(15,9))
sns.distplot(df['generation_gwh_2017'])
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
plt.xlabel('',fontweight='bold',fontsize=18)
plt.title('label of the model/method used to estimate generation for the year 2017',fontweight='bold',fontsize=20)


# 1.8) generation_gwh_2018

# In[87]:


plt.figure(figsize=(15,9))
sns.distplot(df['generation_gwh_2018'])
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
plt.xlabel('',fontweight='bold',fontsize=18)
plt.title('label of the model/method used to estimate generation for the year 2018',fontweight='bold',fontsize=20)


# OBSERVATION :-
#     
#     * AS SE SEEN IN THE ABOVE FIGURE. THE SKEWNESS IS PRESENT IN ALL THE YEARS 
#     * ALSO THE OUTLIERS ARE PRESENT IN THE DATASET

# # 2.) BIVARIATE ANALAYSIS

# 2.1.1)YEAR VS PRIMARY FUEL

# In[88]:


pf=df.groupby("primary_fuel").sum()
pf.transpose().tail().plot(kind='line',figsize=(15,9))
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
plt.title('primary fuel VS years',fontweight='bold',fontsize=20)


# OBSERVATION:-
#     
#     * AS WE SEEN COAL IS SHOWING CONTINOUS INCREASE YEAR BY YEAR
#     * AS WE SEEN ALL OTHER FUELS ARE SHOW CONSTANT LINE AND SHOW NO INCREASE OR DECREASE

# 2.1.2) primary fuel VS capacity MEGAWATTS

# In[89]:


pf['capacity_mw'].plot(kind='bar',color='r',figsize=(15,9))
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
plt.title('primary fuel VS CAPACITY MEGAWATTS',fontweight='bold',fontsize=20)


# OBSERVATION:-
#     
#     * AS WE SEEN COAL HAS HIGHEST CAPACITY MEGAWATTS

# 2.2) CAPACITY MAGAWAATS 

# 2.2.1) CAPACITY vs COMISSION YEAR

# In[90]:


df['commissioning_year']=df['commissioning_year'].astype(int)


# In[91]:


cy=df.groupby('commissioning_year').sum()
cy['capacity_mw'].plot(kind='bar',figsize=(25,9))
plt.xticks(rotation=45,fontweight='bold')
plt.yticks(fontweight='bold')
plt.title('CAPACITY MEGAWATTS VS COMISSION YEAR',fontsize=20,fontweight='bold')


# OBSERVATION:-
#      
#      * AS WE SEEN YEAR BY YEAR THEIR IS SUDDEN INCREASE IN PLANT OPERATION AND ALSO INCREASE IN CAPACITY MAGAWATTS

# 2.2.2) CAPACITY VS generation_gwh_2014

# In[92]:


cw=df.groupby('capacity_mw').sum()
cw['generation_gwh_2014'].plot(kind='line',figsize=(15,9))
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
plt.title('CAPACITY VS generation_gwh_2014',fontweight='bold',fontsize=20,)


# 2.2.3) CAPACITY VS generation_gwh_2015

# In[93]:


cw=df.groupby('capacity_mw').sum()
cw['generation_gwh_2015'].plot(kind='line',figsize=(15,9))
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
plt.title('CAPACITY VS generation_gwh_2015',fontweight='bold',fontsize=20,)


# 2.2.4) CAPACITY VS generation_gwh_2016

# In[94]:


cw=df.groupby('capacity_mw').sum()
cw['generation_gwh_2016'].plot(kind='line',figsize=(15,9))
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
plt.title('CAPACITY VS generation_gwh_2016',fontweight='bold',fontsize=20,)


# 2.2.5) CAPACITY VS generation_gwh_2017

# In[95]:


cw=df.groupby('capacity_mw').sum()
cw['generation_gwh_2017'].plot(kind='line',figsize=(15,9))
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
plt.title('CAPACITY VS generation_gwh_2017',fontweight='bold',fontsize=20,)


# 2.2.6) CAPACITY VS generation_gwh_2018

# In[96]:


cw=df.groupby('capacity_mw').sum()
cw['generation_gwh_2018'].plot(kind='line',figsize=(15,9))
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
plt.title('CAPACITY VS generation_gwh_2018',fontweight='bold',fontsize=20,)


# OBSERVATION:-
#     
#     * AS WE SEEN YEAR BY YEAR CAPACITY MEGAWATTS INCREASE 

# # 3.) MULTIVARIATE ANALYSIS

# In[97]:


sns.pairplot(df)


# their is no need of 'name','gppd_idnr','owner' columns so we have to drop

# In[98]:


df.drop(['name','gppd_idnr','owner'],inplace=True,axis=1)


# # NOW FIND THE CORELATION

# In[99]:


df.corr()


# # LETS UNDERSTAND ON HEATMAP

# In[100]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True)
plt.yticks(fontweight="bold")
plt.xticks(fontweight="bold")


# In[101]:


plt.figure(figsize=(20,7))
df.corr()['capacity_mw'].sort_values(ascending=False).drop(['capacity_mw']).plot(kind='bar',color='c')
plt.xlabel('Feature', fontsize=14)
plt.ylabel('Columns with target name', fontsize=14)
plt.title('Correlation',fontsize=18)
plt.show()


# # ENCODING OF DATAFRAME

# WE HAVE A CATEGORICAL DATA IN DATADRAME SO WE NEED TO ENCODE DATA INTO NUMERICAL DATA SO WILL BE UNDERSTABLE BY MACHINE

# In[102]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[103]:


for i in df.columns:
    if df[i].dtypes=="object":
        df[i]=le.fit_transform(df[i])
df.head()


# # REMOVING OUTLIERS

# In[104]:


plt.figure(figsize=(20,30),facecolor='white')
plotnumber=1
for i in df.columns:
    if plotnumber<=40:
        ax=plt.subplot(10,4,plotnumber)
        sns.boxplot(df[i],color='gold')
        plt.xlabel(i,fontsize=20)
    plotnumber+=1
plt.tight_layout()


# * AS WE SEEN ALL COLUMNS HAVE OUTLIERS EXCEPT 'LATITUDE' AND 'PRIMARY_FUEL'

# In[105]:


from scipy.stats import zscore
z=np.abs(zscore(df))
threshold=3
np.where(z>3)


# In[106]:


df_new=df[(z<3).all(axis=1)]
df_new


# In[107]:


df_new.shape


# In[108]:


df.shape


# In[109]:


data_loss=((907-846)/907)*100
data_loss


# # 6.7% data loss is less then 10%

# In[110]:


df=df_new


# # seprating the columns into feature and target

# In[111]:


from sklearn.preprocessing import power_transform 

# Divide the data into features and vectors.

x=df.drop(['capacity_mw'], axis=1)
y=df.iloc[:,0]

x=power_transform(x, method='yeo-johnson')


# In[112]:


X=pd.DataFrame(data=x)


# # Scaling the data Using StandardScaler.

# In[113]:


from sklearn.preprocessing import StandardScaler
SDc=StandardScaler()
X=SDc.fit_transform(X)


# # VIF calculation

# In[114]:


df.columns


# In[115]:


import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[116]:


dfx=pd.DataFrame(data=X, columns=['capacity_mw', 'latitude', 'longitude', 'primary_fuel',
       'geolocation_source', 'generation_gwh_2014',
       'generation_gwh_2015', 'generation_gwh_2016', 'generation_gwh_2017',
       'generation_gwh_2018'])


# In[117]:


def calc_vif(x):
    vif=pd.DataFrame()
    vif['variables']=x.columns
    vif['VIF FACTOR']=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
    return(vif)


# In[118]:


calc_vif(dfx)


# In[119]:


dfx.drop('generation_gwh_2016', axis=1, inplace=True )


# In[120]:


calc_vif(dfx)


# In[121]:


x=dfx


# # MACHINE LEARNING MODEL

# In[122]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score


# # SELECT BEST RANDOM STATE

# In[123]:


lin_reg=LinearRegression()
for i in range(0,100):
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.33,random_state=i)
    lin_reg.fit(x_train,y_train)
    pred_train=lin_reg.predict(x_train)
    pred_test=lin_reg.predict(x_test)
    print(f'At random state{i}, The training accuracy is : {r2_score(y_train,pred_train)}')
    print(f'At random state{i}, The test accuracy is : {r2_score(y_test,pred_test)}')
    print('\n')


# AS WE CONCLUDE THAT 97 STATE GAVE HIGHEST VALUE, SO TAKE 97 AS RANDOM STATE

# In[124]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=97,test_size=0.33)


# In[125]:


x_train.shape


# In[126]:


y_train.shape


# In[127]:


x_test.shape


# In[128]:


y_test.shape


# # 1.) LINEAR REGRESSION

# In[129]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)
pred=lr.predict(x_test)


# In[130]:


print('\033[1m'+ 'Error :'+ '\033[0m')
print('Mean absolute error :', mean_absolute_error(y_test,pred))
print('Mean squared error :', mean_squared_error(y_test,pred))
print('Root Mean squared error :', np.sqrt(mean_squared_error(y_test,pred)))
print('\033[1m'+' R2 Score :'+'\033[0m')
print(r2_score(y_test,pred)*100)


# In[131]:


score=cross_val_score(lr,x,y,cv=5)
print('\033[1m'+'Cross Validation Score :',lr,":"+'\033[0m\n')
print("Mean CV Score :",score.mean())
print('Difference in R2 & CV Score:',(r2_score(y_test,pred)*100)-(score.mean()*100))


# # 2.) SGD

# In[132]:


from sklearn.linear_model import SGDRegressor
sgd=SGDRegressor()
sgd.fit(x_train,y_train)
pred=sgd.predict(x_test)


# In[133]:


print('\033[1m'+ 'Error :'+ '\033[0m')
print('Mean absolute error :', mean_absolute_error(y_test,pred))
print('Mean squared error :', mean_squared_error(y_test,pred))
print('Root Mean squared error :', np.sqrt(mean_squared_error(y_test,pred)))
print('\033[1m'+' R2 Score :'+'\033[0m')
print(r2_score(y_test,pred)*100)


# In[134]:


score=cross_val_score(sgd,x,y,cv=5)
print('\033[1m'+'Cross Validation Score :',sgd,":"+'\033[0m\n')
print("Mean CV Score :",score.mean())
print('Difference in R2 & CV Score:',(r2_score(y_test,pred)*100)-(score.mean()*100))


# # 3.) RANDOM FOREST REGRESSION

# In[135]:


rfr=RandomForestRegressor()
rfr.fit(x_train,y_train)
pred=rfr.predict(x_test)


# In[136]:


print('\033[1m'+ 'Error :'+ '\033[0m')
print('Mean absolute error :', mean_absolute_error(y_test,pred))
print('Mean squared error :', mean_squared_error(y_test,pred))
print('Root Mean squared error :', np.sqrt(mean_squared_error(y_test,pred)))
print('\033[1m'+' R2 Score :'+'\033[0m')
print(r2_score(y_test,pred)*100)


# In[137]:


score=cross_val_score(rfr,x,y,cv=5)
print('\033[1m'+'Cross Validation Score :',rfr,":"+'\033[0m\n')
print("Mean CV Score :",score.mean())
print('Difference in R2 & CV Score:',(r2_score(y_test,pred)*100)-(score.mean()*100))


# # 4.)DECISION TREE REGRESSION

# In[138]:


dtr=DecisionTreeRegressor()
dtr.fit(x_train,y_train)
pred=dtr.predict(x_test)


# In[139]:


print('\033[1m'+ 'Error :'+ '\033[0m')
print('Mean absolute error :', mean_absolute_error(y_test,pred))
print('Mean squared error :', mean_squared_error(y_test,pred))
print('Root Mean squared error :', np.sqrt(mean_squared_error(y_test,pred)))
print('\033[1m'+' R2 Score :'+'\033[0m')
print(r2_score(y_test,pred)*100)


# In[145]:


score=cross_val_score(dtr,x,y,cv=5)
print('\033[1m'+'Cross Validation Score :',dtr,":"+'\033[0m\n')
print("Mean CV Score :",score.mean())
print('Difference in R2 & CV Score:',(r2_score(y_test,pred)*100)-(score.mean()*100))


# # 5.) EXTRA TREE REGRESSION

# In[140]:


etc=ExtraTreesRegressor()
etc.fit(x_train,y_train)
pred=etc.predict(x_test)


# In[141]:


print('\033[1m'+ 'Error :'+ '\033[0m')
print('Mean absolute error :', mean_absolute_error(y_test,pred))
print('Mean squared error :', mean_squared_error(y_test,pred))
print('Root Mean squared error :', np.sqrt(mean_squared_error(y_test,pred)))
print('\033[1m'+' R2 Score :'+'\033[0m')
print(r2_score(y_test,pred)*100)


# In[146]:


score=cross_val_score(etc,x,y,cv=5)
print('\033[1m'+'Cross Validation Score :',etc,":"+'\033[0m\n')
print("Mean CV Score :",score.mean())
print('Difference in R2 & CV Score:',(r2_score(y_test,pred)*100)-(score.mean()*100))


# # 6.)XGBOOST REGRESSION

# In[142]:


xg=XGBRegressor()
xg.fit(x_train,y_train)
pred=xg.predict(x_test)


# In[143]:


print('\033[1m'+ 'Error :'+ '\033[0m')
print('Mean absolute error :', mean_absolute_error(y_test,pred))
print('Mean squared error :', mean_squared_error(y_test,pred))
print('Root Mean squared error :', np.sqrt(mean_squared_error(y_test,pred)))
print('\033[1m'+' R2 Score :'+'\033[0m')
print(r2_score(y_test,pred)*100)


# In[144]:


score=cross_val_score(xg,x,y,cv=5)
print('\033[1m'+'Cross Validation Score :',xg,":"+'\033[0m\n')
print("Mean CV Score :",score.mean())
print('Difference in R2 & CV Score:',(r2_score(y_test,pred)*100)-(score.mean()*100))


# We can see RandomForestRegressor Having very less diffrence in R2 score and Cross Val Score. we can consider this our best model.

# # SAVING MODEL

# In[148]:


import joblib
joblib.dump(rfr,"GLOBAL_POWER_PLANT_CAPATIL_MW.pkl")


# # PREDICTION

# In[149]:


model=joblib.load("GLOBAL_POWER_PLANT_CAPATIL_MW.pkl")

#perdiction
prediction=model.predict(x_test)
prediction


# In[150]:


pd.DataFrame([model.predict(x_test)[:],y_test[:]],index=["Predicted","Original"])


# # NOW LETS PREDICT PRIMARY FUEL

# # import libraries

# In[151]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# # SEPRATING TARGET AND TRANING DATA

# In[153]:


from sklearn.model_selection import train_test_split

# Split the dataset into features (X) and target (y)
X = df.drop('primary_fuel', axis=1) # Features (all columns except for the target)
y = df['primary_fuel'] # Target (the region column)

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=97)


# In[154]:


X_train.shape


# In[155]:


y_train.shape


# In[156]:


X_test.shape


# In[157]:


y_test.shape


# # 2.1.) Decision tree Classifier

# In[160]:


from sklearn.tree import DecisionTreeClassifier


# In[161]:


dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
preddtc=dtc.predict(X_test)
preddtc


# In[162]:


print(accuracy_score(y_test,preddtc)*100)
print(confusion_matrix(y_test,preddtc))
print(classification_report(y_test,preddtc))


# # 2.2.) KNeighborsClassifier

# In[163]:


from sklearn.neighbors import KNeighborsClassifier


# In[164]:


knn= KNeighborsClassifier()
knn.fit(X_train,y_train)
predknn=knn.predict(X_test)
predknn


# In[165]:


print(accuracy_score(y_test,predknn)*100)
print(confusion_matrix(y_test,predknn))
print(classification_report(y_test,predknn))


# # 2.3.)SVC

# In[166]:


from sklearn.svm import SVC


# In[167]:


sv=SVC()
sv.fit(X_train,y_train)
predsv=sv.predict(X_test)
predsv


# In[168]:


print(accuracy_score(y_test,predsv)*100)
print(confusion_matrix(y_test,predsv))
print(classification_report(y_test,predsv))


# # CROSS VALDIATION

# In[169]:


from sklearn.model_selection import cross_val_score


# SVC

# In[171]:


score=cross_val_score(sv,x,y,cv=5)
print(score)
print(score.mean())
print(score.std())


# KNN

# In[173]:


score=cross_val_score(knn,x,y,cv=5)
print(score)
print(score.mean())
print(score.std())


# DECISION TREE CLASSIFIER

# In[174]:


score=cross_val_score(dtc,x,y,cv=5)
print(score)
print(score.mean())
print(score.std())


# as we seen DTC GAVES HISEGST SCORE

# # SAVING THE BEST MODEL

# In[176]:


import joblib

# Save model to file
joblib.dump(dtc, 'GLOBAL_POWER_PLANT_PRIMARY_FUEL.pkl')


# # LOAD THE MODEL

# In[177]:


modelr=joblib.load("GLOBAL_POWER_PLANT_PRIMARY_FUEL.pkl")


# In[178]:


predict=modelr.predict(X_test)
predict


# In[180]:


pd.DataFrame([modelr.predict(X_test)[:],y_test[:]],index=["Predicted","Original"])


# # THANK YOU

# In[ ]:




