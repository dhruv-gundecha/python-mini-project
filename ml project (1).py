#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import style
plt.style.use('bmh')


# In[4]:


style.use('ggplot')


# In[21]:


data = pd.read_csv(r"C:\Users\dhruv\Documents\data_stocks.csv")
print(data.head(25))


# In[6]:


x = data[['high','low','open','volume']].values
y = data['close'].values


# In[7]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0) 


# In[17]:


reg = LinearRegression()
reg.fit(x_train,y_train)


# In[22]:


y_pred = regressor.predict(x_test)
result = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
result.head(25)


# In[24]:


import math 


# In[ ]:


graph = result.head(20)


# In[ ]:


graph.plot(kind='bar')


# In[25]:


data.shape


# In[26]:


plt.figure(figsize(figsize=(16,8)))
plt.title('Tesla')
plt.xlabel('days')
plt.ylabel('Close Price USD($)')
plt.plot(data['Close'])
plt.show


# In[28]:


data = data[['close']]
data.head(4)


# In[29]:


future_days = 25
data['Prediction']= data [['close']].shift(-future_days)
data.tail(4)


# In[30]:


X = np.array(data.drop(['Prediction'],1))[:-future_days]
print(X)


# In[31]:


y = np.array(data(['Prediction'])[:-future_days]
print(y)


# In[35]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)


# In[33]:


x_future = data.drop(['Prediction'],1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
x_future


# In[34]:


lr = LinearRegression().fit(x_train,y_train)


# In[36]:


lr_prediction = lr.predict(x_future)
print(lr_prediction)


# In[37]:


predictions = lr_prediction
valid = data[X.shape[0]:]
valid['Prediction'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price UDS ($)')
plt.plot(data['close'])
plt.plot(valid[['close','Prediction']])
plt.legend(['Orig','Val','Pred'])
plt.show()


# In[ ]:




