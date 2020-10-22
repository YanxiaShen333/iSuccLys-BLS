#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


def AA_ONE_HOT(AA):
    one_hot_dict = {
        'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'D': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'E': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'F': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'G': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'H': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'I': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'K': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'N': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }
    coding_arr = np.zeros((len(AA), 21), dtype=float)

    for i in range(len(AA)):
        coding_arr[i] = one_hot_dict[AA[i]]

    return coding_arr


# In[48]:


# file_pos = open("Database\succ_train_pos_data_move78.txt")
# file_neg = open("Database\succ_train_neg_data_move78.txt")

file_pos = open("Database\succ_test_pos_data.txt")
file_neg = open("Database\succ_test_neg_data.txt")
data_pos = []
data_neg = []
for line in file_pos.readlines():
    line = line.replace('\n','')
    data_pos.append(line)
for line in file_neg.readlines():
    line = line.replace('\n','')
    data_neg .append(line)
data = data_pos+data_neg
print(len(data_pos))
print(len(data_neg))
print(len(data))
print(data[8])
# AA = AA_ONE_HOT(data[8])
# print(AA)
# print(AA.shape)
# a = AA.flatten().reshape((1,-1))
# print(a)
# print(a.shape)
X = np.zeros((len(data),len(data[0])*21))
print(X.shape)
t = 0
for i in range(len(data)):
    x_tmp = AA_ONE_HOT(data[i]).flatten().reshape((1,-1))
    X[i] = x_tmp
print(X[8])
print(type(X))
print(X.shape)
print(X)
# # X = np.array(X)
# print(X.shape)

# y = []
# for x in data3['Sequences']:
#     y_1=AA_ONE_HOT(x).flatten().reshape((1, 620))
#     y.append(y_1)

# data3['feature1']=y
# data4=np.array(data3)
# data5=data4[:,1:]


# In[49]:


# np.savetxt("Succ_train_one_hot",X)
np.savetxt("Succ_test_one_hot",X)


# In[47]:


data = np.loadtxt("Succ_train_one_hot")
print(data.shape)


# In[50]:


data = np.loadtxt("Succ_test_one_hot")
print(data.shape)

