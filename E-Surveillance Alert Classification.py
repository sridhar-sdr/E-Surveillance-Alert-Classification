
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

# We are tasked with classifying the alert  whether it is Critical, Normal, or Testing which is received from the various sensors. Such as vibration, motion, smoke, fire.

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

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings

warnings.filterwarnings('ignore')


# # Data

# In[87]:


data= pd.read_csv('Master - Ack Time Analysis From April-20 to August-20 - Details.csv',error_bad_lines= False)
data


# ## Showing first 5 rows of the dataset

# In[89]:


data.head(5)


# # Shape

# In[90]:


data.shape


# ## Counting the output values for each categories

# In[103]:


data.Status.value_counts().count


# # Features of the data set

# In[92]:


data.columns


# ## Number of distinct observations 

# In[94]:


data.nunique()


# # Removing the unnecessary features

# In[95]:


p=data.drop(['LOG ID','SENSOR (As of Portal)','DATE','SENSOR NAME (Standard)', 'Month','EVENT DATE AND TIME','Reason'], axis=1)
p


# ## Checking for NaN/null values

# In[105]:


p.isna().sum()


# ## Handling categorical data

# Here we are converting categorical data to numerical data using label encoder

# In[106]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
xm=p.apply(LabelEncoder().fit_transform)
xm


# # Spliting input features

# In[61]:


X=xm.iloc[:,:-1]
X


# # Output feature

# In[132]:


y=xm.iloc[:,8]
y


# In[160]:


feature_names = X.columns.tolist()
feature_names


# Calculate the correlation with y for each feature and collect all correlation values in a list

# In[141]:


list(map(lambda x: np.corrcoef(X[x], y)[0, 1], feature_names))


# In[165]:


cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-3:]].columns.tolist()
cor_feature


# ## Create dataframes from Scores (cor_list) and Features

# In[166]:


dfscores = pd.DataFrame(cor_list)
dfcolumns = pd.DataFrame(X.columns)


# ## Which are the best features?

# In[167]:


featurescores = pd.concat([dfcolumns,dfscores], axis=1)
featurescores
featurescores.columns = ['features','Scores']
featurescores
featurescores.nlargest(10, 'Scores')


# # Split Train and Test data

# In[134]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=5)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## Building machine learing model

# # 1. Logistic Regression

# In[135]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = p

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=5)


model = LogisticRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
print(model.score(X_train, y_train))



# # 2. KNearest Neighbour

# In[68]:


from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
print(knn_classifier.score(X_test, y_test))


# In[73]:



X_ = np.array([1255,619,3,11,6,7872,14,8079])
y_pred =knn_classifier.predict([X_])


X1 = np.array([705,16,3,16,10,2682,1,2748])
y_pred1 =knn_classifier.predict([X1])

print(y_pred)
print(y_pred1)


# # 3. Random Forest

# In[14]:


from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
print(rf_classifier.score(X_test, y_test))


# In[75]:



X_ = np.array([1255,619,3,11,6,7872,14,8079])
y_pred =rf_classifier.predict([X_])
y_pred


# # Conclusion

# In[102]:


print('\n                     Accuracy     Error')
print('                     ----------   --------')
print('Logistic Regression : {:.04}%       {:.04}%'.format( model.score(X_test, y_test)* 100,                                                  100-(model.score(X_test, y_test) * 100)))

print('KNN                 : {:.04}%       {:.04}% '.format(knn_classifier.score(X_test, y_test) * 100,                                                        100-(knn_classifier.score(X_test, y_test) * 100)))

print('Random Forest       : {:.04}%       {:.04}% '.format(rf_classifier.score(X_test, y_test)* 100,                                                           100-(rf_classifier.score(X_test, y_test)* 100)))


# We can choose the Random Forest,KNN model to get the desired output

# Happy learning.....
