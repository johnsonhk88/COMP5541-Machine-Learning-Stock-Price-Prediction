import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision
from torch.autograd import Variable

#from sklearn.metrics import accuracy_score, confusion_matrix,  classification_report


import matplotlib.pyplot as plt

import datetime

import memory_profiler 

import psutil

EnableGPU =False



# Declare Train stock  path 
PreTrainStock1 = 'preprocess_price_train/1_p_price_train.txt'
PreTrainStock2 = 'preprocess_price_train/2_p_price_train.txt'
PreTrainStock3 = 'preprocess_price_train/3_p_price_train.txt'
PreTrainStock4 = 'preprocess_price_train/4_p_price_train.txt'
PreTrainStock5 = 'preprocess_price_train/5_p_price_train.txt'
PreTrainStock6 = 'preprocess_price_train/6_p_price_train.txt'
PreTrainStock7 = 'preprocess_price_train/7_p_price_train.txt'
PreTrainStock8 = 'preprocess_price_train/8_p_price_train.txt'

# Declare Raw stock  path 
RawStock1 = 'raw_price_train/1_r_price_train.csv'
RawStock2 = 'raw_price_train/2_r_price_train.csv'
RawStock3 = 'raw_price_train/3_r_price_train.csv'
RawStock4 = 'raw_price_train/4_r_price_train.csv'
RawStock5 = 'raw_price_train/5_r_price_train.csv'
RawStock6 = 'raw_price_train/6_r_price_train.csv'
RawStock7 = 'raw_price_train/7_r_price_train.csv'
RawStock8 = 'raw_price_train/8_r_price_train.csv'


#declare Move Average day
maDay = [10, 20, 50, 100, 200]


# Load Data from file
trainStock1 = pd.read_csv(PreTrainStock1, header=None, delimiter='\t', 
                     names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])

#load Raw Data 
rawStock1 = pd.read_csv(RawStock1, index_col="Date", parse_dates=True) 
rawStock2 = pd.read_csv(RawStock2, index_col="Date", parse_dates=True) 
rawStock3 = pd.read_csv(RawStock3, index_col="Date", parse_dates=True) 
rawStock4 = pd.read_csv(RawStock4, index_col="Date", parse_dates=True) 
rawStock5 = pd.read_csv(RawStock5, index_col="Date", parse_dates=True) 
rawStock6 = pd.read_csv(RawStock6, index_col="Date", parse_dates=True) 
rawStock7 = pd.read_csv(RawStock7, index_col="Date", parse_dates=True) 
rawStock8 = pd.read_csv(RawStock8, index_col="Date", parse_dates=True) 


preTrainStock1 = pd.read_csv(PreTrainStock1) 
preTrainStock2 = pd.read_csv(PreTrainStock2) 
preTrainStock3 = pd.read_csv(PreTrainStock3) 
preTrainStock4 = pd.read_csv(PreTrainStock4)
preTrainStock5 = pd.read_csv(PreTrainStock5) 
preTrainStock6 = pd.read_csv(PreTrainStock6) 
preTrainStock7 = pd.read_csv(PreTrainStock7) 
preTrainStock8 = pd.read_csv(PreTrainStock8)  


# create RowStock Dictionary 
RawStockList = {}

RawStockList['rawStock1'] = rawStock1
RawStockList['rawStock2'] = rawStock2
RawStockList['rawStock3'] = rawStock3
RawStockList['rawStock4'] = rawStock4
RawStockList['rawStock5'] = rawStock5
RawStockList['rawStock6'] = rawStock6
RawStockList['rawStock7'] = rawStock7
RawStockList['rawStock8'] = rawStock8


# create preTrainStock Dictionary 
PreTrainStockList = {}

PreTrainStockList['preTrainStock1'] = preTrainStock1
PreTrainStockList['preTrainStock2'] = preTrainStock2
PreTrainStockList['preTrainStock3'] = preTrainStock3
PreTrainStockList['preTrainStock4'] = preTrainStock4
PreTrainStockList['preTrainStock5'] = preTrainStock5
PreTrainStockList['preTrainStock6'] = preTrainStock6
PreTrainStockList['preTrainStock7'] = preTrainStock7
PreTrainStockList['preTrainStock8'] = preTrainStock8

def showStockInfo(Stock):
    Stock.info()


def showStockData(Stock):
    print(Stock.head())
    print(Stock.describe())


def calculateMA(Stock):
    for ma in maDay:
        columnName = "MA for %s  days" %(str(ma))
        MaResult=Stock['Adj Close'].rolling(ma).mean()
        print(columnName, MaResult)
        Stock.insert(6, columnName, Stock['Adj Close'].rolling(ma).mean(), True )

def calculateDailyChange(Stock):
    Stock.insert(11, 'Daily Return', Stock['Adj Close'].pct_change(), True )
    

# First calculate the mid prices from the highest and lowest
def CalculateNorm(Stock):
    high_prices = Stock.loc[:,'High'].to_numpy()
    low_prices = Stock.loc[:,'Low'].to_numpy()
    mid_prices = (high_prices+low_prices)/2.0

    return high_prices, low_prices,  mid_prices

def showAllStockInfo(StockList):
    for stockTempKey, stockTempValue in StockList.items():
        print("\n\rShow Stock New Info :", stockTempKey)
        showStockInfo(stockTempValue)

def showAllStockData(StockList):
    for stockTempKey, stockTempValue in StockList.items():
        print("\n\rShow Stock New Info :", stockTempKey)
        showStockData(stockTempValue)


def plotPrice(Stock , name):
    plt.figure(figsize= (20, 10))
    plt.plot(range(Stock.shape[0]),(Stock['Low']+Stock['High'])/2.0, label='Average Price')
    plt.plot(range(Stock.shape[0]),(Stock['Adj Close']), label='Adj')
    plt.plot(range(Stock.shape[0]),(Stock['Open']), label='Open')
    plt.plot(range(Stock.shape[0]),(Stock['Close']), label='Close')
    plt.xticks(range(0,Stock.shape[0],50),Stock.index[::50], rotation=45)
    plt.title('Stock %s : Price' %name)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Mid Price',fontsize=18)
    plt.legend(loc='best')
    plt.show()
    
def plotVolume(Stock , name):
    plt.figure(figsize= (20, 10))
    plt.plot(range(Stock.shape[0]),(Stock['Volume']), label='Volume')
    plt.xticks(range(0,Stock.shape[0],50),Stock.index[::50], rotation=60)
    plt.title('Stock %s : Volume History' %name)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Volume',fontsize=18)
    plt.legend(loc='best')
    plt.show()
    

def plotDailyChange(Stock , name):
    # Then we'll plot the daily return percentage
    plt.title('Stock %s : Dialy Change' %name)
    Stock['Daily Return'].plot(figsize=(20,10), legend=True,linestyle='--',marker='o')
    plt.show()



showAllStockInfo(RawStockList)
showAllStockData(RawStockList)



for stockTempKey, stockTempValue in RawStockList.items(): 
    print("\n\rCalculate MA for Stock :", stockTempKey)
    calculateMA(stockTempValue)
    print("\n\rCalculate Daily Change for Stock :", stockTempKey)
    calculateDailyChange(stockTempValue)
    print("\n\rCalculate High/Low/Average Price :", stockTempKey)
    trainHigPrices , trainLowPrice,  trainMidPrice = CalculateNorm(stockTempValue)
    print("High Data Size :", len(trainHigPrices))
    print("Low Data Size :", len(trainLowPrice))
    print("Average Data Size :", len(trainMidPrice))

showAllStockInfo(RawStockList)

for stockTempKey, stockTempValue in RawStockList.items():
    print("\n\rPlot Price :", stockTempKey)
    plotPrice(stockTempValue, stockTempKey)
    plotVolume(stockTempValue, stockTempKey)
    plotDailyChange(stockTempValue, stockTempKey)


showAllStockInfo(PreTrainStockList)
showAllStockData(PreTrainStockList)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################

        if torch.cuda.is_available() and EnableGPU:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        # Initialize cell state
        if torch.cuda.is_available() and EnableGPU:
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        # One time step
        out, (hn, cn) = self.lstm(x, (h0,c0))
        
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out







'''
train_data = trainMidPrice[:300]
test_data = trainMidPrice[300:]
'''
