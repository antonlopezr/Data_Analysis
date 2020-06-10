import pandas as pd
import numpy as np
import os


# In[2]:


#defining variables
n = 10    
hnumber = 0
inumber = 0


# In[3]:


#crazy stuff happens here
#bins are an integer and the rounded down value of the coordinate. Multiply 100 is needed becaus the values were in mm
binh = pd.read_csv(r'C:\Users\dsanc\Desktop\CFD\CFDcsv.csv')

binh['x bin'] = binh['x'].multiply(100.).apply(np.floor)
binh['y bin'] = binh['y'].multiply(100.).apply(np.floor)
binh['z bin'] = binh['z'].multiply(100.).apply(np.floor)

binh = binh.drop(binh.columns[0],axis = 1)


# In[5]:


#creating the files
for x in range(-20, 20):
    for y in range (-15,15):
        for z in range (-15,15):
            name =  '{}_{}_{}'.format(x,y,z)
            df = pd.DataFrame()
            df.to_csv(r'C:\Users\dsanc\Python\Python36\100bins\{}.csv'.format(name))
            del df


# In[6]:


#writing the file. Goes line by line in the data and finds the file it belongs to by looking at its name
for i in range(len(binh)):

    name = '{}_{}_{}'.format(int(binh.loc[i]['x bin']), int(binh.loc[i]['y bin']), int(binh.loc[i]['z bin']))
    
    
    threedbin = pd.read_csv(r'C:\Users\dsanc\Python\Python36\100bins\{}.csv'.format(name))
    threedbin = threedbin.drop(threedbin.columns[0],axis = 1)
    threedbin = threedbin.append(binh.iloc[i])


    threedbin.to_csv(r'C:\Users\dsanc\Python\Python36\100bins\{}.csv'.format(name))
    print(100*i/len(binh))
    del threedbin

