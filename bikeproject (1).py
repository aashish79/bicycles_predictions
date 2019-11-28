#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train=pd.read_csv("C:/Users/Aashish/Desktop/project/train.csv")
test=pd.read_csv("C:/Users/Aashish/Desktop/project/test.csv")
data = train.append(test, sort = False)
data.head()


# In[4]:


data.dtypes


# In[5]:


data.info()


# In[6]:


data['count'].plot.hist()


# In[7]:


train['count'].plot.hist()


# In[8]:


train['count'].plot.box()


# In[9]:


data['count'].plot.box()


# In[10]:


data.describe()


# In[11]:


data['casual'].plot.hist()


# In[12]:


data['registered'].plot.hist()


# In[13]:


data['season'].value_counts()


# In[14]:


data['holiday'].value_counts()


# In[15]:


data['workingday'].value_counts()


# In[16]:


data['weather'].value_counts()


# In[17]:


data['season'].value_counts().plot.bar()


# In[18]:


sns.barplot(x = 'season', y = 'count', data = data, estimator = np.average)
plt.ylabel('Average Count')
plt.show()


# In[19]:


sns.barplot(x = 'holiday', y = 'count', data = data, estimator = np.average)
plt.ylabel('Average Count')
plt.show()


# In[20]:


sns.barplot(x = 'weather', y = 'count', data =data, estimator = np.average)
plt.ylabel('Average Count')
plt.show()


# In[21]:


data.corr()


# In[22]:


columns=['season', 'holiday', 'workingday', 'weather']
for i in columns:
    data[i] = data[i].apply(lambda x : int(x))


# In[23]:


data.head()


# In[24]:


data['datetime'] = pd.to_datetime(data['datetime'])
data['Hour'] = data['datetime'].apply(lambda x:x.hour)
data['Month'] = data['datetime'].apply(lambda x:x.month)
data['Day of Week'] = data['datetime'].apply(lambda x:x.dayofweek)


# In[ ]:





# In[25]:


sns.lineplot(x = 'Month', y = 'count', data = data, estimator = np.average,hue='weather')
plt.ylabel('Average Count')
plt.show()


# In[26]:


data[data['weather'] == 4]


# In[27]:


fig, axes = plt.subplots(ncols = 2, figsize = (15,5), sharey = True)
sns.pointplot(x = 'Hour', y = 'count', data = data, estimator = np.average, hue = 'workingday', ax = axes[0], palette = 'muted')
sns.pointplot(x = 'Hour', y = 'count', data = data, estimator = np.average, hue = 'holiday', ax = axes[1], palette = 'muted')
ax = [0,1]
for i in ax:
    axes[i].set(ylabel='Average Count')


# In[28]:


data.isnull().sum()


# In[29]:


sns.barplot(x = 'workingday', y = 'count', data = data, estimator = np.average)
plt.ylabel('Average Count')
plt.show()


# In[30]:


data.plot.scatter('temp','count')


# In[31]:


data.plot.scatter('humidity','count')


# In[32]:


data.plot.scatter('windspeed','count')


# In[33]:


plt.figure(figsize = (10,4))
sns.pointplot(x = 'Hour', y = 'count', data = data, estimator=np.average, hue = 'Day of Week', palette='coolwarm')


# In[34]:


sns.pointplot(x = 'Hour', y = 'casual', data = data, estimator = np.average, color = 'blue')
sns.pointplot(x = 'Hour', y = 'registered', data = data, estimator = np.average, color = 'red')
plt.ylabel('Registered')
plt.show()


# In[35]:


data['windspeed'] = data['windspeed'].replace(0, np.nan)
data['windspeed'] = data['windspeed'].fillna(data.groupby('weather')['season'].transform('mean'))


# In[36]:


data.plot.scatter('windspeed','count')


# In[37]:


data['Month_sin'] = data['Month'].apply(lambda x: np.sin((2*np.pi*x)/12))
data['Month_cos'] = data['Month'].apply(lambda x: np.cos((2*np.pi*x)/12))
data['Hour_sin'] = data['Hour'].apply(lambda x: np.sin((2*np.pi*(x+1))/24))
data['Hour_cos'] = data['Hour'].apply(lambda x: np.cos((2*np.pi*(x+1))/24))
data['DayOfWeek_sin'] = data['Day of Week'].apply(lambda x: np.sin((2*np.pi*(x+1))/7))
data['DayOfWeek_cos'] = data['Day of Week'].apply(lambda x: np.cos((2*np.pi*(x+1))/7))


# In[38]:


np.sqrt(data['count']).plot.hist()


# In[39]:


np.power(data['count'],1/3).plot.hist()


# In[40]:


np.power(data['count'],2).plot.hist()


# In[41]:


data['count'] = np.log(data['count'])


# In[53]:


data['count'].plot.hist()


# In[42]:


data_ = pd.get_dummies(data=data, columns=['season', 'holiday', 'workingday', 'weather'])
train_ = data_[pd.notnull(data_['count'])].sort_values(by=["datetime"])
test_ = data_[~pd.notnull(data_['count'])].sort_values(by=["datetime"])


# In[43]:


from sklearn.preprocessing import StandardScaler
cols = ['temp','atemp','humidity', 'windspeed', 'Month_sin', 'Month_cos', 'Hour_sin', 'Hour_cos', 'DayOfWeek_sin','DayOfWeek_cos']
features = data[cols]

#Standard Scaler
scaler = StandardScaler().fit(features.values)
data[cols] = scaler.transform(features.values)


# In[44]:


cols = ['temp','atemp','humidity', 'windspeed', 'Month_sin', 'Month_cos', 'Hour_sin', 'Hour_cos', 'DayOfWeek_sin','DayOfWeek_cos', 'season_1','season_2', 'season_3',
        'holiday_0', 'workingday_0', 'weather_1', 'weather_2', 'weather_3']


# In[45]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[46]:


X = train_[cols]
y = train_['count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[47]:


lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.intercept_)


# In[48]:


pred = lm.predict(X_test)


# In[49]:


sns.scatterplot(x = y_test, y = pred)
plt.xlabel('Actual Count')
plt.ylabel('Predictions')
plt.show()


# In[50]:


print('Rmse:', np.sqrt(metrics.mean_squared_error(np.exp(y_test), np.exp(pred))))


# In[51]:


print('RMSLE:', np.sqrt(metrics.mean_squared_log_error(np.exp(y_test), np.exp(pred))))

