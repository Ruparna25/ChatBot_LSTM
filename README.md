# Bidirectional_LSTM

Bidirectional LSTM Encoder Model -
Bidirectional LSTMs have two recurrent components, a forward recurrent component, and a backward recurrent component. The forward component computes the hidden and cell states, and the backward component computes them by taking the input sequence in a reverse-chronological order, that is, starting from time step Tx to 1. The intuition of using a backward component is that we are creating a way where the network sees future data and learns its weights accordingly. 

![image](https://user-images.githubusercontent.com/29209042/162600791-e7c42b09-0ab0-499c-a35e-9c22222e4cb0.png)

# Dataset –
Cornell-Movie-Dialog Corpus is composed of movie dialog, collected from movies of different genres. Below is a summary of the data.
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

# Relevant file
main.ipynb is the file which contains the model training and ChatBot implementation using Simple Interactive Widgets

