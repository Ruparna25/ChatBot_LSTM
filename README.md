### Overview
A chatbot (Conversational AI) is an automated program that simulates human conversation through text messages, voice chats, or both. It learns to do that based on a lot of inputs, and Natural Language Processing (NLP). The demand of conversational AI is on the rise, due to the increasing demand for AI-powered customer support services, omnichannel deployment, and reduced chatbot development costs. Here I hav tried to create a chatbot using the Encoder - Decoder Bidirectional RNN and also trained another model with Attention Encoder - Decoder model for a higher accuracy.

- Bidirectional LSTM Encoder-Decoder Model

Bidirectional LSTMs have two recurrent components, a forward recurrent component, and a backward recurrent component. The forward component computes the hidden and cell states, and the backward component computes them by taking the input sequence in a reverse-chronological order, that is, starting from time step Tx to 1. The intuition of using a backward component is that we are creating a way where the network sees future data and learns its weights accordingly. 

![image](https://user-images.githubusercontent.com/29209042/162600791-e7c42b09-0ab0-499c-a35e-9c22222e4cb0.png)

- Attention Encoder-Decoder model -

Attention is a mechanism that addresses a limitation of the encoder-decoder architecture on long sequences, and that in general speeds up the learning and lifts the skill of the model no sequence to sequence prediction problems.

<img src='https://github.com/Ruparna25/ChatBot_LSTM/blob/main/images/seq2seq-attention.png' width=400 height=400>
</img>

### Dataset 
The dataset used to train our Cornell-Movie-Dialog Corpus is composed of movie dialog, collected from movies of different genres. Below is a summary of the data.
movie_lines.txt – Contains 304713 number of lines from movies.

- movie_lines.txt
	- contains the actual text of each utterance
	- fields:
		- lineID
		- characterID (who uttered this phrase)
		- movieID
		- character name
		- text of the utterance

movie_conversions.txt – Contains 83097 number of conversations.

- movie_conversations.txt
	- the structure of the conversations
	- fields
		- characterID of the first character involved in the conversation
		- characterID of the second character involved in the conversation
		- movieID of the movie in which the conversation occurred
		- list of the utterances that make the conversation, in chronological 
			order: ['lineID1','lineID2','lineIDN']
			has to be matched with movie_lines.txt to reconstruct the actual content
			
### Data Preprocessing 
The data is in a conversation form, belonging from different genre of movies. I extracted all the converational lines by using the files movie_conversations and movie_lines. Used each conversation to be a question followed by an answer. Below is a sample of the same -

**Sample Conversation data**

*L194 Can we make this quick?  Roxanne Korrine and Andrew Barrett are having an incredibly horrendous public break- up on the quad.  Again.*

*L195 Well, I thought we'd start with pronunciation, if that's okay with you.*

*L196 Not the hacking and gagging and spitting part.  Please.*

*L197 Okay... then how 'bout we try out some French cuisine.  Saturday?  Night?*

**Conversation converted to Sample Question/Answer**

Combining first Lines 1-2 from above

```diff
- Can we make this quick?  Roxanne Korrine and Andrew Barrett are having an incredibly horrendous public break- up on the quad.  Again.
+ Well, I thought we'd start with pronunciation, if that's okay with you.
```

Combining first Lines 2-3 from above

```diff
- Well, I thought we'd start with pronunciation, if that's okay with you.
+ Not the hacking and gagging and spitting part.  Please.
```

Combining first Lines 3-4 from above
  
```diff
- Not the hacking and gagging and spitting part.  Please.
+ Okay... then how 'bout we try out some French cuisine.  Saturday?  Night?
```

As a part of preprocessing some basic data clean up were done, like removing unnecessary characters and format of the words were altered. Then converted them into lowercase and trimmed each converastion pair to limit their length within a threshold value. 

**Building vocabulary**
All the words are present in the trimmed question/answer pair was put in a dictionary along with their frequency in the corpus, this was used for creating embeddings for the encoder. Similaryly another dictionary is created where the key is the number which is index to the word, as value. This is used for interpreting decoder output.

### Design ###
**Model 1 -**

The first model which I trained was Encoder-Decoder Bidrectional LSTM. The embeddings of the question from the question/answer pair was passed to the encoder and the embeddings of answers were passed to as input of the decoder. 

**Model 2 -**

The second model trained was an attention model, same like model 1, the embeddings of the question from the question/answer pair was passed to the encoder and the embeddings of answers were passed to as inout of the decoder. Based on this data and the output of the encoder the context of each word is computed and then fitted to predict the outcome.

**Widgets**

Once the model is trained, a widget setup is done which opens a window where you type in any desired question and it predicts the answer to the asked question. A demo of the same is uploaded in link

### Relevant file
main.ipynb is the file which contains the model training and ChatBot implementation using Simple Interactive Widgets

### To Do

1. Due to hardware resource limitation I had to use a very short vocab size, which leads a pretty poor model accuracy, so I am working on training the model on a high performing system.
2. Also in future I want to try implementing a pretrained model like BERT for training my chatbot.
3. Trying to deploy this through a docker, so as to increase the ease of use.
