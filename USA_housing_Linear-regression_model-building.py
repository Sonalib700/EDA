#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as ps
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[44]:


df= pd.read_csv(r'C:\Users\Admin\Downloads\USA_Housing.csv')
df.head()


# In[45]:


df.info()


# In[50]:


df.describe()


# In[51]:


sns.pairplot(data=df)


# In[55]:


sns.histplot(data = df, x = 'Price', bins=20)


# # Training a Linear Regression Model

# In[58]:


df.columns


# In[59]:


x = df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
y = df['Price']


# In[61]:


from sklearn.model_selection import train_test_split


# In[70]:


x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.4, random_state= 101)


# In[71]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Creating and Training the Model

# In[72]:


from sklearn.linear_model import LinearRegression


# In[73]:


lr= LinearRegression()


# In[74]:


lr.fit(x_train,y_train)


# # Model Evaluation

# In[75]:


lr.coef_


# In[76]:


lr.intercept_


# In[77]:


x_train.columns


# In[80]:


coef_df = pd.DataFrame(data= lr.coef_, index= x_train.columns, columns=['Coefficients'])
coef_df


# # Predictions from our Model

# In[81]:


predictions = lr.predict(x_test)
predictions


# In[83]:


# Visualization of predictions
plt.scatter(predictions, y_test)


# In[85]:


# Visualization of Residuals
sns.histplot(predictions- y_test)


# In[86]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[87]:


print('MAE:', mean_absolute_error(y_test, predictions))
print('MSE:', mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))


# In[ ]:




