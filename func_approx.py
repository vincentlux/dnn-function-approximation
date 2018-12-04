#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
def forward(n, x):
    layer = np.zeros(n)
    # to record sum each time
    record = np.zeros(n)
    if x < 0.5 and x >= 0:
        x_1 = 0
    elif x >= 0.5 and x <= 1:
        x_1 = 1
    else:
        print("out of range")
        return
    layer[0] = x_1
    record[0] = x - x_1/2
    
    if record[0] < 0.25:
        layer[1] = 0
    else:
        layer[1] = 1
    for i in range(1, n-1):
        record[i] = record[i-1] - (layer[i]/(2^i))
        if record[i] < 1/(2**(i+1)):
            layer[i+1] = 0
        else:
            layer[i+1] = 1
            
    # Second dnn
    approx = 0
    # Calculate j summation
    summation = 0
    for i in range(n):
        summation += layer[i]/(2**i)
    # Calculate final value
    for i in range(n):
        # ReLU
        if 0 > 2*(layer[i] - 1) + (1/(2**i))*summation:
            approx += 0
        else:
            approx += 2*(layer[i] - 1) + (1/(2**i))*summation
    # Loss
    loss = np.abs(x**2 - approx)
    
    return approx, loss
    


# In[31]:


import math
import random
plot_approx = []
plot_loss = []
plot_x2 = []
rand_list = []
for i in range(10000):
    rand = random.uniform(0, 1)
    rand_list.append(rand)
    approx, loss = forward(40, rand)
    plot_approx.append(approx)
    plot_loss.append(loss)
    plot_x2.append(rand**2)


# In[35]:


import numpy as np
import matplotlib.pyplot as plt
plt.plot(rand_list, plot_loss)


# In[19]:


loss


# In[ ]:




