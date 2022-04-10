# ChatBot_LSTM
Creating a simple Chatbot using LSTM model
Bidirectional LSTM Encoder Model -
Bidirectional LSTMs have two recurrent components, a forward recurrent component, and a backward recurrent component. The forward component computes the hidden and cell states, and the backward component computes them by taking the input sequence in a reverse-chronological order, that is, starting from time step Tx to 1. The intuition of using a backward component is that we are creating a way where the network sees future data and learns its weights accordingly. 

![image](https://user-images.githubusercontent.com/29209042/162600791-e7c42b09-0ab0-499c-a35e-9c22222e4cb0.png)

Dataset –
Cornell-Movie-Dialog Corpus is composed of movie dialog, collected from movies of different genres. Below is a summary of the data.
movie_lines.txt – Contains xx number of lines from movies
movie_conversions.txt – Contains xx number of conversations  
