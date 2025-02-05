#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv(r'C:\Users\Admin\Downloads\archive (13)\car_evaluation.csv')
df.head()


# In[4]:


df.shape


# In[5]:


col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names


# In[6]:


df.head()


# In[7]:


df.info()


# In[9]:


col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
for col in col_names :
    print(df[col].value_counts())


# In[10]:


df['class'].value_counts()


# In[11]:


df.isnull().sum()


# In[12]:


x = df.drop(['class'],axis=1)
y = df['class']


# In[13]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)


# In[21]:


from sklearn.preprocessing import OrdinalEncoder


# In[22]:


label_encoder = OrdinalEncoder()


# In[23]:


x_train = label_encoder.fit_transform(x_train)
x_test = label_encoder.transform(x_test)


# In[27]:


from sklearn.tree import DecisionTreeClassifier


# In[28]:


clf_gini = DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=0)


# In[29]:


clf_gini.fit(x_train,y_train)


# In[32]:


y_pred = clf_gini.predict(x_test)


# In[33]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[34]:


clf_en = DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)


# In[35]:


clf_en.fit(x_train,y_train)


# In[39]:


y_pred_en = clf_en.predict(x_test)


# In[40]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_en))


# In[ ]:




