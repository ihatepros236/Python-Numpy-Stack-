#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[5]:


df=pd.read_csv('C:/Users/Muhammad Ahmad/Downloads/airfoil_self_noise.dat', sep='\t', header=None)


# In[6]:


df.head()


# In[8]:


df.info(


# In[10]:


data=df[[0,1,2,3,4]].values


# In[11]:


target=df[5].values


# In[12]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, Y_train, Y_test =train_test_split(data, target, test_size=0.33)


# In[17]:


from sklearn.linear_model import LinearRegression 


# In[18]:


model=LinearRegression()


# In[19]:


model.fit(X_train,Y_train)


# In[20]:


model.score(X_test,Y_test)


# In[21]:


model.score(X_train,Y_train)


# In[22]:


predictions=model.predict(X_test)


# In[23]:


predictions


# In[24]:


from sklearn.ensemble import RandomForestRegressor


# In[30]:


model2=RandomForestRegressor()


# In[31]:


model2.fit(X_train,Y_train)


# In[32]:


model2.score(X_test,Y_test)


# In[ ]:




