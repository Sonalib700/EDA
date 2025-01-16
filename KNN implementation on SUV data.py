#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


# In[5]:


data = pd.read_csv(r'C:\Users\Admin\Downloads\suv_data.csv')
data.sample(5)


# In[7]:


data['Gender'] = data['Gender'].replace('Female',0)
data['Gender'] = data['Gender'].replace('Male',1)


# In[8]:


data.head()


# In[9]:


data.drop('User ID', axis=1, inplace=True)


# In[12]:


x = data.iloc[:,0:3]
y = data['Purchased']


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state= 42)


# # without scaling

# In[14]:


model = KNeighborsClassifier(n_neighbors=3)
model


# In[15]:


model.fit(x_train,y_train)


# In[16]:


y_pred = model.predict(x_test)
y_pred


# In[17]:


y_test


# In[18]:


print(classification_report(y_test,y_pred))


# In[19]:


cm = confusion_matrix(y_test,y_pred)
cm


# # prediction with log_reg

# In[25]:


from sklearn.linear_model  import LogisticRegression
model_log=LogisticRegression()
model_log.fit(x_train,y_train)
pred_log=model_log.predict(x_test)
pred_log


# In[26]:


y_test


# In[27]:


accuracy_score(y_test,pred_log)


# # Scaling 

# In[29]:


scaler = StandardScaler()
scaler.fit(x)


# In[31]:


scaled_features = scaler.transform(x)
scaled_features.round(2)


# In[33]:


data.head()


# In[35]:


data.feat = pd.DataFrame(scaled_features, columns=data.columns[:-1])
data.feat.head()


# In[36]:


x_train, x_test, y_train, y_test = train_test_split(scaled_features,data['Purchased'],test_size=0.25, random_state=42)


# In[37]:


knn = KNeighborsClassifier(n_neighbors=1)
knn


# In[38]:


knn.fit(x_train,y_train)


# In[40]:


y_pred = knn.predict(x_test)
y_pred


# In[41]:


y_test


# In[42]:


print(classification_report(y_test,y_pred))


# In[43]:


print(confusion_matrix(y_test,y_pred))


# In[44]:


error_rate=[]
for i in range(1,5):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[45]:


plt.figure(figsize=(10,5))
plt.plot(range(1,5),error_rate,color='blue')
plt.title('Error_rate vs K-value')
plt.xlabel('K')
plt.ylabel('Error rate')


# In[46]:


knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)
pred = knn.predict(x_test)

print('WITH K=5')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:




