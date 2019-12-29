#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
from sklearn.datasets import load_breast_cancer


# In[13]:


data=load_breast_cancer() 


# In[14]:


type(data)


# In[15]:


data.keys()


# In[16]:


data.data


# In[17]:


data.data.shape


# In[18]:


data.target


# In[19]:


data.target_names


# data.target.shape

# In[ ]:





# In[21]:


data.feature_names


# In[22]:


data.feature_names.shape


# In[23]:


from sklearn.model_selection import train_test_split 


# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(data.data,data.target,test_size=0.33)


# In[25]:


from sklearn.ensemble import RandomForestClassifier


# In[29]:


model= RandomForestClassifier()
model.fit(X_train,Y_train)


# 

# model.fit(X_train,Y_train)
# 

# In[30]:


model.score(X_train,Y_train)


# In[31]:


model.score(X_test,Y_test)


# In[33]:


predictions= model.predict(X_test)


# In[34]:


predictions


# In[37]:


N=len(Y_test)


# In[38]:


N


# In[40]:


np.sum(predictions==Y_test)/N


# In[41]:


from sklearn.neural_network import MLPClassifier


# In[43]:


from sklearn.preprocessing import StandardScaler


# In[44]:


scaler=StandardScaler()
X_train2=scaler.fit_transform(X_train)


# In[58]:


X_test2=scaler.fit_transform(X_test)
model1=MLPClassifier(max_iter=500)


# 

# In[62]:


model1.fit(X_train2,Y_train)


# 

# In[63]:


model1.score(X_test2, Y_test)


# In[ ]:




