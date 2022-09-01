#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math

def euclidean_distance(point1, point2):
  '''
  Args:
    point1: type array two values [x, y]
    point2: type array two values [x, y]
  Returns:
    Distance of two points
  Descriptions:
    Calculate Euclidian distance of two points in Euclidian space
  '''

  return round(math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2), 2)


# In[2]:


def length(curve1,curve2):
    acc_length = []
    for i in range(0, len(curve1)):
        acc_length.append(euclidean_distance(curve1[i], curve2[i]))
    return acc_length


# In[3]:


import scipy.interpolate as interp
import numpy as np
def check_similarity(f1,f2): 
    # returns similarity. if the value is closer to 0, both curves are more similar, if close to1 graphs are nt similar 
    a= [len(f1), len(f2)]
    p = max(a)
    a=[f1,f2]
    new_f2=[]
    for i in range(len(a)):
        arr2_interp = interp.interp1d(np.arange(a[i].size),a[i])
        new_f2.append(arr2_interp(np.linspace(0,a[i].size-1,p)))
    y1= new_f2[0]
    y2= new_f2[1]
    if (np.mean(y1)> np.mean(y2)):
        y1= y1- (np.mean(y1)- np.mean(y2))
    else:
        y2 =  y2- (np.mean(y2)- np.mean(y1))
    c1= y1/max(y1)
    c2= y2/max(y2)
    x= np.linspace(0, len(c1)-1, num=len(c1))
    shape1=np.column_stack((x, c1))
    shape2=np.column_stack((x, c2))
    stemp= length(shape1,shape2)
    similarity= np.mean(stemp)+np.std(stemp)
    return similarity


# In[4]:


import math
import matplotlib.pyplot as plt
# plt.figure(figsize=(15,20)) 
# c= ['red', 'blue', 'magenta', 'green', 
#                    'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen', 'maroon', 'purple', 'olive', 'teal', 'chocolate',
#                     'crimson', 'rosybrown', 'brown', 'salmon', 'coral', 'orangered', 'sienna', 
#                     'tan', 'gold', 'wheat', 'lawngreen', 'lightgreen', 'turquoise', 'cadetblue', 'steelblue', 'navy', 
#                     'violet', 'blueviolet']

# counts=[2,3]
# y1= fiber[counts[0]]
# y2= fiber[counts[1]]
x = np.linspace(1, -1, num=200)


y1= 2*x**2 +2
y2= -2*x**3 +x +2
newt=[y1,y2]
for i in range(len( newt)):
#     if i in check:
    plt.plot(newt[i])#, c[i])#, label= l[counts[i]])
similarity= check_similarity(y1,y2)
# plt.legend(loc='best')
plt.title(f'Shape similarity is: {similarity}', fontsize=14, fontweight='bold')
plt.show()


# In[ ]:




