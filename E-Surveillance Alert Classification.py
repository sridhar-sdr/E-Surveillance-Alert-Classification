
# coding: utf-8

# #                                      E-Surveillance Alert Classification
# 

# # Business Problem

# Description:
# 
# Prevent break-ins before they occur using IoT security cameras with built-in computer vision capabilities, reducing the need for human intervention. Automated security to safeguard and alert against threats from intrusion or fire using multi-capability sensors such as vibration, motion, smoke, fire, etc.  Ensure the safety of both monetary and intellectual assets with round-the-clock surveillance and controlled access management.
# 
# 

# # Problem Statement

# We are tasked with classifying the alert whether it is Critical, Normal, or Testing which is received from the various sensors. Such as vibration, motion, smoke, fire, Panic, shutter(Door sensor).
# 
# True_Normal= Testing the sensors(Smoke, Panic, fire), Smoke Alert due to AC maintenance / UPS maintenance, Cash loading, Pressing the panic switches unknowingly by bank staffs… 
# 
# True_Critical= Smoke, Fire, Network Connection error, PIR, Panic(Fire, Theft Attempt, in ATM and Bank)
# 
# False_Normal= Sensor Malfunctioning (keep on getting the alert), No activity from PIR, shutter sensors.
# 
# False_Critical = No activity, But, Alert is received from the critical sensor. Such as Smoke, fire, PIR(Motion detection sensor)
# 

# # Real world/Business Objectives and Constraints

# 1.The cost of a mis-classification can be very high.
# 
# 2.No strict latency concerns.

# # Mapping the real world problem to an ML problem

# # Type of Machine Leaning Problem

# Supervised Learning:
# 
# It is a Multi classification problem, for a given sensor data we need to classify if it is critical, Normal, or Testing

# # Train and Test Construction

# We build train and test by randomly splitting in the ratio of 70:30 or 80:20 whatever we choose as we have sufficient points to work with.

# # Importing Necessary Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings

warnings.filterwarnings('ignore')


# # Data

# In[ ]:


data1= pd.read_csv('Master - Ack Time Analysis From April-20 to August-20 - Details.csv')
data=data1.drop(['Branch'],axis=1)
data


# ## Showing first 5 rows of the dataset

# In[ ]:


data.head(5)


# In[ ]:


#Diiferent types of sensor are used in the bank


# In[133]:


sensors_type=data['SENSOR NAME (Standard)'].drop_duplicates()
sensors_type


# # Shape

# In[4]:


data.shape


# ## Counting the output values for each categories

# In[5]:


data.Status.value_counts().count


# # Features of the data set

# In[6]:


data.columns


# ## Number of distinct observations 

# In[13]:


data.nunique()


# # Removing the Unnecessary Columns

# In[ ]:


p=data.drop(['LOG ID','SENSOR (As of Portal)','DATE','SENSOR NAME (Standard)', 'Month','EVENT DATE AND TIME','Reason'], axis=1)
p


# ## Checking for NaN/null values

# In[141]:


p.isna().sum()


# ## Handling categorical data

# Here we are converting categorical data to numerical data using label encoder

# In[142]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
xm=p.apply(LabelEncoder().fit_transform)
xm


# # Spliting input features

# In[143]:


X=xm.iloc[:,:-1]
X


# # Output feature

# In[144]:


y=xm.iloc[:,7]
y


# In[145]:


#columns to list
feature_names = X.columns.tolist()
feature_names


# # Feature Importance

# In[146]:


#importing the ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt


# In[147]:


model = ExtraTreesRegressor()


# In[148]:


model.fit(X,y)

print(model.feature_importances_)


# # Which are the best features?

# In[149]:


#Top 5 important features
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# Calculate the correlation with y for each feature 

# In[150]:


xm.corr()


# # Heat Map

# In[151]:


#get correlations of each features in dataset
corrmat = xm.corr()
corr_features = corrmat.index
plt.figure(figsize=(25,25))
#plot heat map

c=sns.heatmap(xm[corr_features].corr(),annot=True,cmap="RdYlGn")


# df.pivot(index='Business', columns='Date', values='Value')
# 

# # Split Train and Test data

# In[152]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=5)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## Building machine learing model

# # 1. Logistic Regression

# In[153]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = p

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=5)


model = LogisticRegression()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))
print(model.score(X_train, y_train))



# Logistic Regression with L1 regularization

# In[158]:


LR = LogisticRegression( solver='liblinear',penalty = 'l1', C = 0.01) 


# In[159]:


LR.fit(X_train, y_train)


# In[162]:


print(LR.score(X_test, y_test))
print(LR.score(X_train, y_train))


# # 2. GradientBoostingClassifier

# In[167]:


from sklearn.ensemble import GradientBoostingClassifier


# In[177]:


clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)


# In[168]:


print(clf.score(X_test, y_test))
print(clf.score(X_train, y_train))


# # 3. KNearest Neighbour

# In[173]:


from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
print(knn_classifier.score(X_test, y_test))


# In[181]:



X_= np.array([1255,3,11,6,7872,14,8079])
y_pred =knn_classifier.predict([X_])


X1 = np.array([807,0,13,10,2968,19,3045])
y_pred1 =knn_classifier.predict([X1])

print(y_pred)
print(y_pred1)


# # 4. Random Forest

# In[174]:


from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
print(rf_classifier.score(X_test, y_test))


# In[182]:



X_= np.array([807,0,13,10,2968,19,3045])
y_pred =rf_classifier.predict([X_])
y_pred


# # 5. Support Vector Machine

# In[ ]:


#from sklearn import svm


# In[ ]:


#Sv_Classifier= svm.SVC()


# In[ ]:


#Sv_Classifier.fit(X_train, y_train)


# In[ ]:


#Sv_predictions = Sv_Classifier.predict(X_test)


# In[ ]:


#print(Sv_classifier.score(X_test, y_test))


# # Conclusion

# In[183]:


print('\n                     Accuracy     Error')
print('                     ----------   --------')
print('Logistic Regression : {:.04}%       {:.04}%'.format( model.score(X_test, y_test)* 100,                                                  100-(model.score(X_test, y_test) * 100)))

print('KNN                 : {:.04}%       {:.04}% '.format(knn_classifier.score(X_test, y_test) * 100,                                                        100-(knn_classifier.score(X_test, y_test) * 100)))

print('Random Forest       : {:.04}%       {:.04}% '.format(rf_classifier.score(X_test, y_test)* 100,                                                           100-(rf_classifier.score(X_test, y_test)* 100)))
print('Gradient Boosting    : {:.04}%      {:.04}% '.format(clf.score(X_test, y_test)* 100,                                                           100-(clf.score(X_test, y_test)* 100)))
#print('Support Vector Machine : {:.04}%     {:.04}% '.format(Sv_Classifer.score(X_test, y_test)* 100,\
                                                        #   100-(Sv_classifier.score(X_test, y_test)* 100)))


# We can choose the Random Forest,KNN model to get the desired output

# Happy learning.....
