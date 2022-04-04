#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow
import nltk
import json


# Readine the lines and the conversation texts to extract text for training the chatbot

# In[33]:


# Open the movie_lines file and econding used for this file is utf-8, read function helps in reading the file and 
#split function formats the data
lines = open('movie_lines.txt',encoding='utf-8', errors='ignore').read().split('\n')

#Reading the conversations file
conv_lines = open('movie_conversations.txt',encoding='utf-8',errors='ignore').read().split('\n')


# In[91]:


#Extracting the id and the conversation from the movie_lines text file.
# Step 1 - Splitting the file on the indicator '+++$+++'
# Step 2 - Extracting the 5th section of each line as that depicts  the conversation and storing the convesation against the ID
#          present in the first section of the sentence
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# Conversation lines data 
# Step 1: Splitting each line on the identifier
# Step 2: Extracting the last segement of the sentence and then replace quotation marks and in between spaces
conv=[]
for line in conv_lines:
    _conv = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conv.append(_conv.split(','))


# In[128]:


#A sample conversation -
#u16 +++$+++ u25 +++$+++ m1 +++$+++ ['L2256', 'L2257', 'L2258', 'L2259', 'L2260'] this is the 250th row from the input
for i in conv[0]:
    print(i, id2line[i])


# In[135]:


# Taking a pair of conversation and breaking it into input and response pair, ny drawing the lines from the id2line dict
pairs=[]
for cnv in conv:
    for i in range(len(cnv)-1):
        inp=id2line[cnv[i]].strip()
        res=id2line[cnv[i+1]].strip()
    if inp and res:
        pairs.append([inp,res])


# In[134]:





# In[120]:


ans[:10]


# In[ ]:




