# -*- coding: utf-8 -*-
from collections import Counter, defaultdict
import os
import json
import torch
import math
import string
import datetime
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
#print(torch.cuda.is_available())

filenames = []
datalines = []
tempWordList = []
vocabularyList = []
resultEpoch = []
resultLoss = []  

InputFolderPath = ''

# Declare Train stock  path 
PreTrainStock1 = InputFolderPath+'preprocess_price_train/1_p_price_train.txt'
PreTrainStock2 = InputFolderPath+'preprocess_price_train/2_p_price_train.txt'
PreTrainStock3 = InputFolderPath+'preprocess_price_train/3_p_price_train.txt'
PreTrainStock4 = InputFolderPath+'preprocess_price_train/4_p_price_train.txt'
PreTrainStock5 = InputFolderPath+'preprocess_price_train/5_p_price_train.txt'
PreTrainStock6 = InputFolderPath+'preprocess_price_train/6_p_price_train.txt'
PreTrainStock7 = InputFolderPath+'preprocess_price_train/7_p_price_train.txt'
PreTrainStock8 = InputFolderPath+'preprocess_price_train/8_p_price_train.txt'

preTrainStock1 = pd.read_csv(PreTrainStock1, header=None, delimiter='\t', 
                     names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],  index_col="Date") 
preTrainStock2 = pd.read_csv(PreTrainStock2, header=None, delimiter='\t', 
                     names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],  index_col="Date") 
preTrainStock3 = pd.read_csv(PreTrainStock3, header=None, delimiter='\t', 
                     names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],  index_col="Date") 
preTrainStock4 = pd.read_csv(PreTrainStock4, header=None, delimiter='\t', 
                     names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],  index_col="Date")
preTrainStock5 = pd.read_csv(PreTrainStock5, header=None, delimiter='\t', 
                     names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],  index_col="Date") 
preTrainStock6 = pd.read_csv(PreTrainStock6, header=None, delimiter='\t', 
                     names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],  index_col="Date") 
preTrainStock7 = pd.read_csv(PreTrainStock7, header=None, delimiter='\t', 
                     names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],  index_col="Date") 
preTrainStock8 = pd.read_csv(PreTrainStock8, header=None, delimiter='\t', 
                     names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"], index_col="Date") 

def unique_list(inList1):   
    list_set = set(inList1) 
    unique_list = (list(list_set))
    return unique_list

tweetTrainDataFrame = pd.DataFrame(columns = ['Date', 'Dataline', 'Value'])
folderpath = InputFolderPath+'tweet_train/1_tweet_train/'
filenames = os.listdir(folderpath)
for filename in filenames:
    with open(folderpath+filename,'r') as fileInput:
        for line in fileInput:
            datalines.append([json.loads(line),datetime.datetime.strptime(json.loads(line)['created_at'], '%a %b %d %H:%M:%S %z %Y').strftime("%Y-%m-%d")])

datalines = sorted(datalines,key=lambda x:x[1])
previous_document_date_time = ''
for jsonline in datalines:
    document_date_time = datetime.datetime.strptime(jsonline[0]['created_at'], '%a %b %d %H:%M:%S %z %Y').strftime("%Y-%m-%d")
    if document_date_time in preTrainStock1.index:
        if previous_document_date_time == '':
            previous_document_date_time = document_date_time
        if previous_document_date_time != document_date_time:
            tempWordList = unique_list(tempWordList)
            tweetTrainDataFrame.loc[len(tweetTrainDataFrame)] = [previous_document_date_time, tempWordList, preTrainStock1.loc[previous_document_date_time,"Adj Close"]]
            previous_document_date_time = document_date_time
            tempWordList = []

        for word in jsonline[0]['text']:
            vocabularyList.append(word)
            tempWordList.append(word)
#last loop write       
if document_date_time in preTrainStock1.index:        
    tweetTrainDataFrame.loc[len(tweetTrainDataFrame)] = [previous_document_date_time, jsonline[0]['text'], preTrainStock1.loc[previous_document_date_time,"Adj Close"]]     

#vocabularyList = [''.join(c for c in s if c not in string.punctuation) for s in vocabularyList]
#vocabularyList = [s for s in vocabularyList if s]
vocabularyTops = Counter(vocabularyList).most_common(2000)
vocabularyTopsWords = [item[0] for item in vocabularyTops]

sentence_vectors = []
for index, row in tweetTrainDataFrame.iterrows():
    sent_vec = []
    for token in vocabularyTopsWords:
        if token in tweetTrainDataFrame.iloc[index,1]:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    sentence_vectors.append(sent_vec)
#sentence_vectors_array = np.asarray(sentence_vectors)

'''
inputX = sentence_vectors
inputY = tweetTrainDataFrame.iloc[:,2]
clf = svm.SVR()
clf.fit(inputX, inputY)
clf.predict([inputX[465]])
'''
'''
inputX = sentence_vectors[:400]
inputY = tweetTrainDataFrame.iloc[:400,2]
clf = svm.SVR()
clf.fit(inputX, inputY)
clf.predict([sentence_vectors[470]])

scores = cross_val_score(clf, inputX, inputY, cv=10)
print(scores)
'''


inputX = sentence_vectors
inputY = tweetTrainDataFrame.iloc[:,2]

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

inputs = Variable(torch.Tensor(inputX).float())
labels = Variable(torch.LongTensor(inputY).float())

print("inputs shape: ", inputs.shape)
print("labels shape: ", labels.shape)

# Reshaping 3 dimension data
inputs = np.reshape(inputs, (inputs.shape[0], 1, inputs.shape[1]))
print("inputs Shape after reshape: ", inputs.shape)


# hyperparameters
input_size=2000
output_size=1
hidden_dim=5
batch_size = 1
n_layers=5
lr=0.1

loss_val = []

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        
        self.hidden_dim=hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        #self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)
    
        #r_out, hidden = self.rnn(x, hidden)
        r_out, hidden = self.lstm(x, hidden)
        hidden_size = hidden[-1].size(-1)  # for LSTM
        r_out = r_out.view(-1, hidden_size) # for LSTM
        output = self.fc(r_out)
        return output, hidden

# start main program  

# instantiate an RNN
rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)

# MSE loss and Adam optimizer with a learning rate of 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)



final_hidden = None
# train the RNN
def train(rnn, n_steps, print_every):
    
    hidden = None     
    
    for batch_i, step in enumerate(range(n_steps)):
        prediction, hidden = rnn(inputs, hidden)
        final_hidden = hidden

        ## Representing Memory ##
        # make a new variable for hidden and detach the hidden state from its history
        # this way, we don't backpropagate through the entire history
        #hidden = hidden.data
        hidden = hidden # for LSTM

        loss = criterion(prediction, labels)
        loss_val.append(loss.item())

        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # for LSTM
        #loss.backward()
        optimizer.step()

        if batch_i%print_every == 0 or batch_i == n_steps-1:
            print("Epoch: {0}/{1}".format(batch_i+1, n_steps))
            print('Loss: ', loss.item())
            print(prediction.data)
        resultEpoch.append(batch_i)
        resultLoss.append(loss.item())
    
    return rnn

train_result = train(rnn, 50, 1)



plt.figure(figsize=(12,8))
plt.plot(resultEpoch, resultLoss)
plt.xlabel('Step (Training)')
plt.ylabel('Loss (%)')
plt.title('Loss Vs Number of Step')
plt.show()




