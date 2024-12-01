#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib


# In[61]:


df = pd.read_parquet(r'C:\Users\Admin\Downloads\yellow_tripdata_2022-05.parquet', engine='pyarrow')
df.head()


# In[62]:


df.shape


# In[63]:


df.describe()


# In[64]:


df.info()


# In[65]:


df.isna().sum()


# In[66]:


df[df.duplicated()]


# In[67]:


df.duplicated().sum()


# In[73]:


#Finding passenger counts 
round(df['passenger_count'].value_counts(normalize = True),6)


# In[74]:


# Finidng payment type distribution
df['payment_type'].value_counts()


# In[76]:


# Remove rows where 'payment_type' is 3.0 or 3.5
df = df[~df['payment_type'].isin([0,3.0, 3.5,4.0])]

# Verify the result
df['payment_type'].value_counts()


# In[78]:


# filtering for payment type 1 and 2
df = df[df['payment_type']<3]

# filtering for passenger count from 1 to 2
df = df[(df['passenger_count']>0)&(df['passenger_count']<6)]


# In[80]:


df['payment_type'].replace([1,2],['Card','Cash'],inplace = True)
df.head()


# In[81]:


# Type_Casting
df.dtypes


# In[82]:


df['VendorID'] = df['VendorID'].astype(str)
df['passenger_count'] = df['passenger_count'].astype(int)
df['RatecodeID'] = df['RatecodeID'].astype(str)


# In[83]:


df.dtypes


# In[84]:


#REMOVAL OF NULL VALUES
print('% of null values ' , round(129524/len(df),2))


# In[85]:


null_values = df.dropna(inplace=True)


# In[86]:


print("Shape of DataFrame after dropping NaN values:", df.shape)


# In[87]:


df.isna().sum()


# In[88]:


# REmoving duplicates 
df.drop_duplicates(inplace = True)


# In[89]:


df.head()


# In[90]:


df.drop(columns=['mta_tax', 'tolls_amount', 'improvement_surcharge', 'congestion_surcharge'], inplace=True)


# In[93]:


df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
df['duration'] = df['duration'].dt.total_seconds()/60


# In[96]:


# filtering the records for only positive values
df = df[df['fare_amount']>0]
df = df[df['trip_distance']>0]
df = df[df['duration']>0]
df.head()


# In[99]:


df = df[['passenger_count','payment_type','fare_amount','trip_distance','duration']]
df


# In[103]:


round(df.describe(),3)


# In[104]:


plt.hist(df['fare_amount'])


# In[106]:


#OUtliers present hence using IQR to remove outliers:

for col in ['fare_amount','trip_distance','duration']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    IQR = q3-q1
    
    lower_bound = q1-1.5*IQR
    upper_bound = q3+1.5*IQR
    
    df = df[(df[col]>= lower_bound) & (df[col]<= upper_bound)]


# In[108]:


df


# In[112]:


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title('Distribution of fare amount')
plt.hist(df[df['payment_type']=='Card']['fare_amount'],histtype = 'barstacked',bins=20,label='Card',edgecolor='k')
plt.hist(df[df['payment_type']== 'Cash']['fare_amount'],histtype = 'barstacked',bins=20,label='Cash',edgecolor='k')
plt.legend()

plt.subplot(1,2,2)
plt.title('Distribution of trip distance')
plt.hist(df[df['payment_type'] == 'Card']['trip_distance'],histtype='barstacked',bins=20,label='Card',edgecolor='k')
plt.hist(df[df['payment_type']== 'Cash']['trip_distance'],histtype='barstacked',bins=20,label='Cash',edgecolor='k')
plt.legend()
plt.show()


# In[114]:


# calculating the mean and standard deviation group by on payment type
df.groupby('payment_type').agg({'fare_amount' : ['mean','std'], 'trip_distance' : ['mean','std']})


# In[118]:


#Preference of payment type
plt.figure(figsize=(10,5))
plt.title('Preference of payment type')
plt.pie(df['payment_type'].value_counts(normalize=True), labels = df['payment_type'].value_counts().index, autopct = '%1.1f%%',shadow=True, startangle=90)
plt.show()


# In[129]:


# calculating the total passenger count distribution based on the different payment type
passenger_count = df.groupby(['payment_type','passenger_count'])[['passenger_count']].count()

passenger_count.rename(columns = {'passenger_count':'count'},inplace = True)
passenger_count.reset_index(inplace = True)
passenger_count['perc'] = (passenger_count['count']/passenger_count['count'].sum()) * 100
passenger_count


# In[130]:


# creating a new empty dataframe to store the distribution of each payment type (useful for the visualization)
df1 = pd.DataFrame(columns = ['payment_type',1,2,3,4,5])
df1


# In[131]:


df1['payment_type'] = ['Card','Cash']
df1.iloc[0,1:] = passenger_count.iloc[:5,-1]
df1.iloc[1,1:] = passenger_count.iloc[5:,-1]
df1


# In[ ]:


fig, ax = plt.subplots(figsize=(12,5))
df1.plot(x='payment_type', kind='barh', stacked = True, ax=ax)

# Add percentage text
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.text(x + width / 2,
            y + height / 2,
            '{:.0f}%'.format(width),
            horizontalalignment='center',
            verticalalignment='center')


# # Hypothesis Testing

# In[138]:


import statsmodels.api as sm


# In[139]:


fig = sm.qqplot(df['fare_amount'], line='45')
plt.show()


# In[ ]:


# H0 = There is no significance difference between customers who uses card and customers who uses cash
# H1 = There is difference between customers who uses card and customers who uses cash


# In[140]:


from scipy import stats


# In[143]:


credit_card = df[df['payment_type'] == 'Card']['fare_amount']
cash = df[df['payment_type'] == 'Cash']['fare_amount']

# performing t test on both the different sample
t_stat, p_value = stats.ttest_ind(a=credit_card, b=cash, equal_var=False)
print(f"T-statistic: {t_stat}, P-value: {p_value}")


# In[144]:


# comparing the p value with the significance of 5% or 0.05
if p_value < 0.05:
    print("\nReject the null hypothesis")
else:
    print("\nAccept the null hypothesis")


# In[ ]:




