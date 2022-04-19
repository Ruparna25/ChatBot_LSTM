#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow
import nltk
import json
import codecs
import csv
import regex as re
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import keras
from keras.layers import SimpleRNN
from keras.preprocessing.sequence import pad_sequences
import nltk


# Readine the lines and the conversation texts to extract text for training the chatbot

# In[3]:


# Open the movie_lines file and econding used for this file is utf-8, read function helps in reading the file and 
#split function formats the data
lines = open('movie_lines.txt',encoding='utf-8', errors='ignore').read().split('\n')

#Reading the conversations file
conv_lines = open('movie_conversations.txt',encoding='utf-8',errors='ignore').read().split('\n')


# In[4]:


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


# In[5]:


#A sample conversation -
#u16 +++$+++ u25 +++$+++ m1 +++$+++ ['L2256', 'L2257', 'L2258', 'L2259', 'L2260'] this is the 250th row from the input
for i in conv[0]:
    print(i, id2line[i])


# In[6]:


# Taking a pair of conversation and breaking it into input and response pair, ny drawing the lines from the id2line dict
pairs=[]
for cnv in conv:
    for i in range(len(cnv)-1):
        inp=id2line[cnv[i]].strip()
        res=id2line[cnv[i+1]].strip()
        if inp and res:
            pairs.append([inp,res])


# In[7]:


pairs[0:10]


# In[8]:


#Storing inputs and response pair in a text file
file_name='conv_formatted.txt'
delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))
with open(file_name,'w',encoding='utf-8') as out_file:
    writer=csv.writer(out_file,delimiter=delimiter, lineterminator='\n')
    for pair in pairs:
        writer.writerow(pair)


# In[9]:


#Reading the formatted conversation file
line_fmt = open(file_name,encoding='utf-8').read().strip().split('\n')       #Strip removes the leading and trailing characters
len(line_fmt)


# In[10]:


#Some basic formatting of the data
#Converting the data into lower-case, trim and remove all non-letter characters
def NormalizeText(s):
    s=s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

#filterPairs - Stripping sentences into smaller ones by setting a threshold limit on number of words, if either the input
#or the response is less than threshold length, then it is added to the list of valid pairs, else skipped
#'Can we make this quick?  Roxanne Korrine and Andrew Barrett are having an incredibly horrendous public break- up on the quad.
#Again.\tWell, I thought we'd start with pronunciation, if that's okay with you.'
#the above sentence pair is not considered as valid when the max_length is 10 as both the input & response's length > 10
def filterPairs(pairs,max_length):
    valid_pair=[]
    for pair in pairs:
        inp, resp = pair[0].split(' '),pair[1].split(' ')
        if len(inp) < max_length and len(resp) < max_length:
            valid_pair.append(pair)
    print(f'load total {len(valid_pair)} pairs with length <= max_length (10)')
    return valid_pair


def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|]", "", text)
#     text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = " ".join(text.split())
    return text

pairs=[[NormalizeText(clean_text(s)) for s in l.split('\t')] for l in line_fmt]
valid_pairs=filterPairs(pairs,20)
print(valid_pairs[0:3])


# In[11]:


r = np.random.randint(1,len(valid_pairs))

for i in range(r, r+3):
    print(valid_pairs[i])
    print(valid_pairs[i+1])
    print()


# In[12]:


#Sampling only 30000 pairs of input and response to train the model
num_samples = 30000
sampled_input = []
sampled_resp = []
for pair in valid_pairs:
    
    sampled_input.append(pair[0])
    sampled_resp.append(pair[1])
sampled_input=sampled_input[:30000]
sampled_resp=sampled_resp[:30000]


# In[13]:


sampled_input_tok = [nltk.word_tokenize(word) for word in sampled_input]
sampled_resp_tok = [nltk.word_tokenize(word) for word in sampled_resp]


# In[14]:


data_size=len(sampled_input)

training_input = sampled_input_tok[:round(data_size*(80/100))]

#Reversing the input sequence for better performance - as the first word received by the decoder would be the last encoded word
#example 'you are asking me out . that is so cute . that is your name again ?'
#. esaelp . trap gnittips dna gniggag dna gnikcah eht ton
training_input = [tr_input[::-1] for tr_input in training_input]
training_output = sampled_resp_tok[:round(data_size*(80/100))]

validation_input = sampled_input_tok[round(data_size*(80/100)):]
validation_input = [val_input[::-1] for val_input in validation_input]
validation_output = sampled_resp_tok[round(data_size*(80/100)):]

print('training size', len(training_input))
print('validation size', len(validation_input))


# In[15]:


#Creating a dictionary of the words present in the sampled input and response
vocab={}
for inp in sampled_input_tok:
    for word in inp:
        if word not in vocab:
            vocab[word]=1
        else:
            vocab[word] +=1
        
for resp in sampled_resp_tok:
    for word in inp:
        if word not in vocab:
            vocab[word]=1
        else:
            vocab[word] +=1


# In[16]:


#Removing words which appeared 7 or less number of times in the vocabulary
threshold = 7
count=0
for k,v in vocab.items():
    if v>= threshold:
        count+=1
        
print('Size of actual vocabulary:',len(vocab))
print('Size of vocab to be used:',count)


# In[17]:


sos_token = 1
pad_token = 0

# the word num should be set as 2 as 1 is allocated for sos_token and 0 is allocated for pad token
word_num = 2
encoding = {}
decoding = {1:'SOS'}

for word, count in vocab.items():
    if count >= threshold:
        encoding[word]=word_num
        decoding[word_num]=word
        word_num+=1
        
print('No of vocab used:',word_num)


# In[18]:


#include unknown token for words not in dictionary
decoding[len(encoding)+2] = '<UNK>'
encoding['<UNK>'] = len(encoding)+2


# In[19]:


#Increased dict_size by 1 for the unknown word
dict_size = word_num+1
dict_size


# In[20]:


def transform(encoding, data, vector_size):
    
    transformed_data = np.zeros(shape=(len(data),vector_size))
    for i in range(len(data)):
        #Trim all the sentences which contains more than the max allowed no of words in a sentence
        for j in range(min(len(data[i]),vector_size)):
            try:
                transformed_data[i][j]=encoding[data[i][j]]
            except:
                transformed_data[i][j]=encoding['<UNK>']
    return transformed_data


# In[21]:


INPUT_LENGTH = 15
OUTPUT_LENGTH = 15
encoding_input_data = transform(encoding, training_input, vector_size=INPUT_LENGTH)
encoding_resp_data = transform(encoding, training_output, vector_size=OUTPUT_LENGTH)

print('Encoded Training Input:',encoding_input_data.shape)
print('Encoded Training Output:',encoding_resp_data.shape)


# In[22]:


encoding_input_data[:3]


# In[23]:


encoding_valid_input_data = transform(encoding, validation_input, vector_size=INPUT_LENGTH)
encoding_valid_resp_data = transform(encoding,validation_output,vector_size=OUTPUT_LENGTH)

print('Encoded Training Input:',encoding_valid_input_data.shape)
print('Encoded Training Output:',encoding_valid_resp_data.shape)


# Attention Model -
# 1. Embedding Layer for drawing insight from the text data
# 2. LSTM layer for Encoder
# 3. LSTM layer for Decoder
# 4. Attention Model 
# 5. Dense Model

# x = np.arange(10).reshape(1, 5, 2)
# print(x)
# y = np.arange(10, 20).reshape(1, 2, 5)
# print(y)
# keras.layers.dot([x,y],axes=(2,1))

# In[24]:


encoder_inputs = keras.layers.Input(shape=(INPUT_LENGTH,))
decoder_inputs = keras.layers.Input(shape=(OUTPUT_LENGTH,))


# In[25]:


with tf.device('/GPU:0'):
    encoder_embeddings = keras.layers.Embedding(dict_size,128,input_length=INPUT_LENGTH,mask_zero=True,)(encoder_inputs)
    
    encoder = keras.layers.LSTM(512,return_sequences=True,unroll=True)(encoder_embeddings)
    encoder_last = encoder[:,-1,:]
    
    print('encoder',encoder)
    print('encoder_last',encoder_last)
    
    decoder_embeddings = keras.layers.Embedding(dict_size,128,input_length=OUTPUT_LENGTH,mask_zero=True,)(decoder_inputs)
    decoder = keras.layers.LSTM(512,return_sequences=True,unroll=True)(decoder_embeddings,initial_state=[encoder_last,encoder_last])
    
    print(decoder)


# In[26]:


with tf.device('/GPU:0'):
    attention=keras.layers.dot([encoder,decoder],axes=[2,2])
    attention=keras.layers.Activation('softmax',name='attention')(attention)
    print('attention:',attention)
    
    context=keras.layers.dot([attention,encoder],axes=[2,1])
    print('context:',context)
    
    decoder_combined_context=keras.layers.concatenate([context,decoder])
    print('decoder_combined_context:',decoder_combined_context)
    
    output=keras.layers.TimeDistributed(keras.layers.Dense(512,activation='tanh'))(decoder_combined_context)
    output=keras.layers.TimeDistributed(keras.layers.Dense(dict_size,activation='softmax'))(output)
    print('output:',output)


# In[27]:


with tf.device('/GPU:0'):
    
    model = keras.Model(inputs=[encoder_inputs,decoder_inputs],outputs=[output])
    model.compile(optimizer='adam',loss='binary_crossentropy')
    model.summary()


# In[28]:


training_encoder_input = encoding_input_data
training_decoder_input = np.zeros_like(encoding_resp_data)
#Copy all characters except the last the one and move it from 1st pos
training_decoder_input[:, 1:] = encoding_resp_data[:,:-1] 
training_decoder_input[:,0] = sos_token
training_decoder_output = np.eye(dict_size)[encoding_resp_data.astype('int')]


# In[29]:


validation_encoder_input = encoding_valid_input_data
validation_decoder_input = np.zeros_like(encoding_valid_resp_data)
#Copy all characters except the last the one and move it from 1st pos
validation_decoder_input[:, 1:] = encoding_valid_resp_data[:,:-1] 
validation_decoder_input[:,0] = sos_token
validation_decoder_output = np.eye(dict_size)[encoding_valid_resp_data.astype('int')]


# In[30]:


#Training the model
with tf.device('/GPU:0'):
    model.fit(x=[training_encoder_input,training_decoder_input],y=[training_decoder_output],
              validation_data=([validation_encoder_input,validation_decoder_input],[validation_decoder_output]),
              batch_size=64,
              epochs=100)


# In[48]:


model.save('C:/Users/manju/OneDrive/Desktop/Ruparna/ChatBot Using RNN/movie_dialogue_attention.h5')


# Prediction Function

# In[44]:


def predictions(sentence):
    
    input_text = NormalizeText(clean_text(sentence))
    input_tok = [nltk.word_tokenize(input_text)]

    encoder_input_data=transform(encoding,input_tok,15)
    decoder_input_data=np.zeros(shape=(len(encoder_input_data),OUTPUT_LENGTH))
    decoder_input_data[:,0]=sos_token
    for i in range(1,OUTPUT_LENGTH):
        output=model.predict([encoder_input_data,decoder_input_data]).argmax(axis=2)
        decoder_input_data[:,i]=output[:,i]
    
    return output

def decode(target_data,decoding):
    text=''
    for i in range(0,OUTPUT_LENGTH):
        idx=int(target_data[:,i])
        if idx==0:
            return text
            break
        else:
            text+=' '+decoding[idx]
    return text


# In[45]:


sentence = 'hi! Are you there'
output=predictions(sentence)
decode(output,decoding)


# In[46]:


#from ipywidgets import interact, interactive, fixed, interact_manual

def process_req(ques):
    output = predictions(ques)
    decoder_resp = decode(output,decoding)
    return decoder_resp

#for i in range(5):
#    interact(process_req, ques=input())


# In[47]:





# In[ ]:




