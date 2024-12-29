#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r'C:\Users\Admin\Downloads\Ecommerce Customers.csv')
df


# In[4]:


df.info()           #Checking null values


# In[8]:


df.describe().T       #checking statistical measures (.T means transpose of the data)


# In[10]:


#Use seaborn to create a jointplot to compare the Time on Website and app and Yearly Amount Spent columns. Does the correlation make sense?


sns.jointplot(data=df, x='Time on Website', y='Yearly Amount Spent')


# In[11]:


sns.jointplot(data=df, x='Time on App', y='Yearly Amount Spent')


# In[13]:


df.columns


# In[15]:


# Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.
sns.jointplot(data=df, x='Time on App', y='Length of Membership', kind ='hex')


# In[16]:


# Let's explore these types of relationships across the entire data set. Use pairplot to recreate the plot below.(Don't worry about the the colors)

sns.pairplot(df)


# In[18]:


sns.lmplot(data=df, x='Length of Membership', y='Yearly Amount Spent')


# In[ ]:


# Training and Testing Data


# In[20]:


df.columns


# In[21]:


# separate features and labels
x = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = df['Yearly Amount Spent']


# In[22]:


from sklearn.model_selection import train_test_split


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 101)


# In[26]:


x_train.shape , y_train.shape , x_test.shape , y_test.shape


# In[ ]:


# Training the Model


# In[27]:


from sklearn.linear_model import LinearRegression


# In[28]:


lr = LinearRegression()


# In[29]:


lr.fit(x_train,y_train)


# In[ ]:


# Print out the coefficients of the model


# In[32]:


lr.coef_


# In[34]:


# converting it in dataframe 
df_coef = pd.DataFrame(data= lr.coef_, index= x_train.columns , columns=['Coefficient'])
df_coef


# In[35]:


df_coef.sort_values(by='Coefficient',ascending=False)


# In[ ]:


# Now that we have fit our model, let's evaluate its performance by predicting off the test values!
# Create a scatterplot of the real test values versus the predicted values.


# In[36]:


predictions = lr.predict(x_test)
predictions


# In[37]:


plt.scatter(predictions,y_test)


# In[ ]:


# Evaluating the model
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).


# In[38]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score


# In[40]:


print('MAE : {}'.format(mean_absolute_error(y_test,predictions)))
print('MSE : {}'.format(mean_squared_error(y_test,predictions)))
print('RMSE : {}'.format(np.sqrt(mean_squared_error(y_test,predictions))))
print('Explained_variance_score: {}'.format(explained_variance_score(y_test,predictions)))


# In[ ]:


# You should have gotten a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data.


# In[44]:


# Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().
sns.histplot(y_test-predictions, kde=True, bins=35)


# In[45]:


# It seem a good Residual Historgram as the shape is pretty normally distributed.


# In[ ]:




