#!/usr/bin/env python
# coding: utf-8

# # BREAST CANCER PREDICTION

# ### DATA SET- https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

# In[2]:


import numpy as np                   
import pandas as pd 
import seaborn as sns  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# # DATA ANALYSIS

# In[3]:


df = pd.read_csv('/home/anmol/Downloads/breast-cancer-wisconsin-data/data.csv')
df.head()


# # ATTRIBUTE INFORMATION:
# 
# 1) ID number
# 2) Diagnosis (M = malignant, B = benign)
# 
# '3-32'.Ten real-valued features are computed for each cell nucleus:
# 
# a) radius (mean of distances from center to points on the perimeter)
# 
# b) texture (standard deviation of gray-scale values)
# 
# c) perimeter
# 
# d) area
# 
# e) smoothness (local variation in radius lengths)
# 
# f) compactness (perimeter^2 / area - 1.0)
# 
# g). concavity (severity of concave portions of the contour)
# 
# h). concave points (number of concave portions of the contour)
# 
# i). symmetry
# 
# j). fractal dimension ("coastline approximation" - 1)
# 
# 
# Attributes (3-32) are divided into three parts each conataining ten features:
# 
# 
# Mean (3-13),
# 
# Standard Error(13-23)
# 
# Worst(23-32)
# 

# # Observations from data frame 
# 
# 1) There is an id that cannot be used for classificaiton 
# 
# 2) Diagnosis is our class label 
# 
# 3) Unnamed: 32 feature includes NaN so we do not need it

# In[5]:


class_label=df['diagnosis']
class_label


# In[6]:


list = ['Unnamed: 32','id','diagnosis']
x= df.drop(list,axis = 1 )     # x is our clean df
x.head()


# ### class label is stored in class_label field  and rest data is in table named x

# In[7]:


plot1 = sns.countplot(class_label,label="Count")      
B, M = class_label.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)


# In[8]:


x.describe()


# ### max value of area_mean = 2501 and compactness_mean is 0.16 there is so much difference in these value area value will overpower value of smoothness so we need to standardize or normalize our data

# # Chosing between Standardisation and Normalisation

# In[14]:


table = x
data=x
table_std = (table - table.mean()) / (table.std())  #standardisation
table_norm=(data-data.max())/(data.max()-data.min()) #normalisation


# In[15]:


table = pd.concat([class_label,table_norm.iloc[:,0:10]],axis=1)
table = pd.melt(table,id_vars="diagnosis",var_name="features",value_name='value')
plt.figure(figsize=(20,20))

sns.violinplot(y="features", x="value", hue="diagnosis", data=table ,palette="Set2",split=True, inner="quart")
plt.xticks(rotation=0)


# In[16]:


data = pd.concat([class_label,table_std.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",var_name="features",value_name='value')
plt.figure(figsize=(20,20))
sns.violinplot(y="features", x="value", hue="diagnosis", data=data,palette="Set2",split=True, inner="quart")


# #### As on standardisation graphs are showing more uniformity , as compared to normalisation so I am analyzing data on standardizing data

# In[17]:


data = pd.concat([class_label,table_std.iloc[:,0:30]],axis=1)
data = pd.melt(data,id_vars="diagnosis",var_name="features",value_name='value')
plt.figure(figsize=(20,40))
sns.violinplot(y="features", x="value", hue="diagnosis", data=data,palette="Set2",split=True, inner="quart")


# In[49]:


data = pd.concat([class_label,table_std.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",var_name="features",value_name='value')
plt.figure(figsize=(10,10))
sns.boxplot(y="features", x="value", hue="diagnosis", data=data,palette="Set2")

plt.xticks(rotation=90)


# ## Observation
# 
# For example, in texture_mean feature, median of the Malignant and Benign looks like separated so it can be good for classification.
# However, in fractal_dimension_mean feature, median of the Malignant and Benign does not looks like separated so it does not gives good information for classification.
# 

# # Variance in Data

# In[18]:



data = pd.concat([class_label,table_std.iloc[:,0:30]],axis=1)
data = pd.melt(data,id_vars="diagnosis",var_name="features",value_name='value')
plt.figure(figsize=(30,60))

sns.swarmplot(y="features", x="value", hue="diagnosis", data=data, palette="muted")



# ## Observation
# area_mean , concavity_mean in last swarm plot looks like malignant and benign are seprated not totaly but mostly. Hovewer, smoothness_se , fractal_dimension_worst in swarm plot  looks like malignant and benign are mixed so it is hard to classfy while using this feature.

# # Correlation

# In[9]:


f,ax = plt.subplots(figsize=(25, 25))
sns.heatmap(x.corr(), annot=True, fmt= '.1f',ax=ax)


# #### it can be seen in map heat figure radius_mean, perimeter_mean and area_mean are correlated with each other so we will use only area_mean.

# #### Compactness_mean, concavity_mean and concave points_mean are correlated with each other.Therefore I only choose concavity_mean. Apart from these, radius_se, perimeter_se and area_se are correlated and I only use area_se. radius_worst, perimeter_worst and area_worst are correlated so I use area_worst. Compactness_worst, concavity_worst and concave points_worst so I use concavity_worst. Compactness_se, concavity_se and concave points_se so I use concavity_se. texture_mean and texture_worst are correlated and I use texture_mean. area_worst and area_mean are correlated, I use area_mean.

# In[12]:


remove = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']
x_mod= x.drop(remove,axis = 1 )        
x_mod.head()


# In[13]:


f,ax = plt.subplots(figsize=(19, 19))
sns.heatmap(x_mod.corr(), annot=True, fmt= '.1f',ax=ax)


# # Decision Tree Classifier

# In[20]:


x_train, x_test, y_train, y_test = train_test_split(x_mod, class_label, test_size=0.4)
model=tree.DecisionTreeClassifier()  
dtc = model.fit(x_train,y_train)
ac = accuracy_score(y_test, model.predict(x_test))
print('Accuracy is: ',ac*100,'%')
cv_results = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(cv_results)))


# # Naive Bayes Classifier

# In[22]:


x_train, x_test, y_train, y_test = train_test_split(x_mod, class_label, test_size=0.4, random_state=1)
model =GaussianNB()      
nbc = model.fit(x_train,y_train)
ac = accuracy_score(y_test,model.predict(x_test))
print('Accuracy is: ',ac*100 ,'%')

cv_results = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(cv_results)))


# # Logistic Regression

# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x_mod, class_label, test_size=0.4, random_state=1)
model =LogisticRegression()      
lrm = model.fit(x_train,y_train)
ac = accuracy_score(y_test,model.predict(x_test))
print('Accuracy is: ',ac*100 ,'%')

cv_results = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(cv_results)))


# # Random Forest Classifier

# In[24]:


x_train, x_test, y_train, y_test = train_test_split(x_mod, class_label, test_size=0.4, random_state=1)
model =RandomForestClassifier()      
rfc = model.fit(x_train,y_train)
ac = accuracy_score(y_test,model.predict(x_test))
print('Accuracy is: ',ac*100 ,'%')

cv_results = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(cv_results)))


# # Univariate feature selection

# In[26]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

select_feature = SelectKBest(chi2, k=5).fit(x_train, y_train)


# In[27]:


print('Score list:', select_feature.scores_)
print('Feature list:', x_train.columns)


# ### Best 5 feature to classify is that area_mean, area_se, texture_mean, concavity_worst and concavity_mean.

# In[28]:


train = select_feature.transform(x_train)
test = select_feature.transform(x_test)


# # Decision Tree on Best 5 features

# In[29]:


clf_2 = tree.DecisionTreeClassifier()      
clr_2 = clf_2.fit(train,y_train)
ac_2 = accuracy_score(y_test,clf_2.predict(test))
print('Accuracy is: ',ac_2*100,'%')


# # Naive Bayes on best 5 features

# In[30]:


clf_2 = GaussianNB()      
clr_2 = clf_2.fit(train,y_train)
ac_2 = accuracy_score(y_test,clf_2.predict(test))
print('Accuracy is: ',ac_2*100,'%')


# # Logistic Regression on best 5 features

# In[31]:


clf_2 = LogisticRegression()      
clr_2 = clf_2.fit(train,y_train)
ac_2 = accuracy_score(y_test,clf_2.predict(test))
print('Accuracy is: ',ac_2*100,'%')


# # Random Forest Classifier on best 5 features

# In[32]:


clf_2 = RandomForestClassifier()      
clr_2 = clf_2.fit(train,y_train)
ac_2 = accuracy_score(y_test,clf_2.predict(test))
print('Accuracy is: ',ac_2*100,'%')


# # Recursive feature elimination (RFE) with random forest

# In[33]:


from sklearn.feature_selection import RFE       
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=5, step=30)
rfe = rfe.fit(x_train, y_train)
print('Chosen best 5 feature by rfe:',x_train.columns[rfe.support_])


# In[ ]:





# # Recursive feature elimination with cross validation and decision tree classification

# In[34]:


from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=tree.DecisionTreeClassifier(), cv=5,scoring='accuracy')  
rfecv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])


# In[35]:


import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# # Recursive feature elimination with cross validation and Logistic Regression

# In[36]:


from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=LogisticRegression(), cv=5,scoring='accuracy')  
rfecv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])


# In[37]:


import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# # Recursive feature elimination with cross validation and random forest classification

# In[38]:


from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=RandomForestClassifier(), cv=5,scoring='accuracy')  
rfecv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])


# In[39]:


# Plot number of features VS. cross-validation scores
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# In[ ]:




