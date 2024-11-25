#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[19]:


data = pd.read_excel(r'C:\Users\Admin\Downloads\amazon_sales.csv.xlsx')


# In[21]:


data.head(5)


# In[23]:


data.info()


# In[24]:


data.tail()


# In[26]:


#Checking null values
pd.isnull(data).sum()


# In[28]:


data.shape


# In[29]:


#Remove null values
data.dropna(inplace=True)


# In[30]:


data.shape


# In[31]:


data.isnull().sum()


# In[32]:


#change data type
data['ship-postal-code'] = data['ship-postal-code'].astype(int)


# In[36]:


#Verifying the changes
data['ship-postal-code'].dtype


# In[38]:


data['Date'] = pd.to_datetime (data['Date'])


# In[39]:


data['Date'].dtype


# In[40]:


data.describe ()


# In[41]:


data.describe (include='object')


# In[42]:


data.columns


# In[45]:


data[['Amount','Qty']].describe()


# In[50]:


# countplot of sizes
ax= sns.countplot(x='Size',data=data)
for i in ax.containers:
    ax.bar_label(i)


# In[51]:


data 


# In[53]:


# plotting courier status vs status 
plt.figure(figsize=(10,5))
sns.countplot(data=data, x='Courier Status', hue='Status')


# In[54]:


data.info()


# In[55]:


data['Category']= data['Category'].astype(str)


# In[61]:


plt.hist(data=data, x='Category', bins= 20, edgecolor = 'Purple')
plt.xticks(rotation=90)
plt.show()


# In[70]:


# checking B2B data
check_b2b = data['B2B'].value_counts()
plt.pie(check_b2b, labels= check_b2b.index, autopct = '%1.1f%%')
plt.show()


# In[75]:


# Category plotting with size
plt.figure(figsize=(10,5))
plt.scatter(x='Size',y='Category',data=data,color='Red')
plt.show()


# In[ ]:




