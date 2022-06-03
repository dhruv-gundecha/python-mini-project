#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[ ]:





# In[28]:


data = pd.read_csv(r"C:\Users\dhruv\Documents\data_stocks.csv")


# In[29]:


x = data[['high','low','open','volume']].values
y = data['close'].values


# In[32]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0) 


# In[ ]:





# In[ ]:




