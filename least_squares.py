
# coding: utf-8

# In[1]:

import random
import numpy as np


# In[2]:

# generate x and Y values for y=2x+5+E
l = 26
x0 = np.array([1 for _ in range(l)])
x1 = np.array([x/5 for x in range(l)]) # from 0 to 5, step 0.2
X = np.array([[x0[i], x1[i]] for i in range(len(x1))])
Y = np.array([2*x+5+random.random()-1 for x in x1])

print(X)
print(Y)


# In[3]:

C = np.dot(np.transpose(X),X)
G = np.linalg.inv(C)

B = np.dot(np.dot(G, np.transpose(X)),Y)
print(B) # should be somewhere close to 5 and 2


# In[ ]:



