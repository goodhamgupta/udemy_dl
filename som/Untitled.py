
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('Credit_Card_Applications.csv')


# In[3]:


df.head()


# In[4]:


X = df.iloc[:, :-1].values


# In[5]:


y = df.iloc[:, -1].values


# ### Feature scaling

# In[6]:


from sklearn.preprocessing import MinMaxScaler


# In[7]:


scaler = MinMaxScaler(feature_range=(0,1))


# In[8]:


X_scaled = scaler.fit_transform(X)


# ### Training

# In[9]:


from minisom import MiniSom


# In[10]:


som = MiniSom(10, 10, input_len=15)


# In[11]:


som.random_weights_init(X)


# In[12]:


som.train_batch(X, num_iteration=100)


# ### Visualizing results

# In[13]:


from pylab import bone, pcolor, colorbar, plot, show


# In[15]:


bone()


# In[16]:


pcolor(som.distance_map().T)


# In[17]:


colorbar()


# - Red cirles => Didn't get approval. Green circles => Got approval

# In[18]:


markers = ['o', 's']
colors = ['r', 'g']


# In[21]:


for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] +0.5, w[1] +0.5,
        markers[y[i]],
        markeredgecolor=colors[y[i]],
        markerfacecolor=None,
        markersize=10,
        markeredgewidth=2)


# In[22]:


show()


# ### Finding the frauds

# In[25]:


mappings = som.win_map(X_scaled)


# In[29]:


mappings


# In[37]:


frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis=0)


# In[38]:


frauds


# In[39]:


scaler.inverse_transform(frauds)


# In[40]:


import pdb
pdb.set_trace()

