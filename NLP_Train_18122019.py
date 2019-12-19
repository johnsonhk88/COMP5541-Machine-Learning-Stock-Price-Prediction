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
#print(torch.cuda.is_available())

filenames = []
datalines = []
tempWordList = []
vocabularyList = []

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
#print('\n\rData Content After sort:' , datalines)
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

print("Vocablary List :", vocabularyList)
print("\n\rTemp Word List :", tempWordList)
#last loop write       
if document_date_time in preTrainStock1.index:        
    tweetTrainDataFrame.loc[len(tweetTrainDataFrame)] = [previous_document_date_time, jsonline[0]['text'], preTrainStock1.loc[previous_document_date_time,"Adj Close"]]     

print('Tweet Train Data frame : ', tweetTrainDataFrame)
#vocabularyList = [''.join(c for c in s if c not in string.punctuation) for s in vocabularyList]
#vocabularyList = [s for s in vocabularyList if s]
vocabularyTops = Counter(vocabularyList).most_common(1000)
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
print('Input X Type : ', type(inputX) , 'Data : ', inputX)
print('Input Y Type : ', type(inputY) , 'Data : ', inputY)

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

inputs = Variable(torch.Tensor(inputX))
labels = Variable(torch.LongTensor(inputY))

# hyperparameters
seq_len = 1000      # |hihell| == 6, equivalent to time step
input_size = 471   # one-hot size
batch_size = 1   # one sentence per batch
num_layers = 2   # one-layer rnn
num_classes = 5  # predicting 5 distinct character
hidden_size = 5  # output from the RNN


class RNN(nn.Module):
    """
    The RNN model will be a RNN followed by a linear layer,
    i.e. a fully-connected layer
    """
    def __init__(self, seq_len, num_classes, input_size, hidden_size, num_layers):
        super().__init__()
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first = True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # assuming batch_first = True for RNN cells
        batch_size = x.size(0)
        hidden = self._init_hidden(batch_size)
        x = x.view(batch_size, self.seq_len, self.input_size)

        # apart from the output, rnn also gives us the hidden
        # cell, this gives us the opportunity to pass it to
        # the next cell if needed; we won't be needing it here
        # because the nn.RNN already computed all the time steps
        # for us. rnn_out will of size [batch_size, seq_len, hidden_size]
        rnn_out, _ = self.rnn(x, hidden)
        linear_out = self.linear(rnn_out.view(-1, hidden_size))
        return linear_out

    def _init_hidden(self, batch_size):
        """
        Initialize hidden cell states, assuming
        batch_first = True for RNN cells
        """
        return Variable(torch.zeros(
            batch_size, self.num_layers, self.hidden_size))


# Set loss, optimizer and the RNN model
torch.manual_seed(777)
rnn = RNN(seq_len, num_classes, input_size, hidden_size, num_layers)
print('network architecture:\n', rnn)

# train the model
num_epochs = 15
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr = 0.1)
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()
    outputs = rnn(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # check the current predicted string
    # max gives the maximum value and its
    # corresponding index, we will only
    # be needing the index
    _, idx = outputs.max(dim = 1)
    idx = idx.data.numpy()
    print('epoch: {}, loss: {:1.3f}'.format(epoch, loss.item()))


































