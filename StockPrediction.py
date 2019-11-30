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

RawStockList = [] 
RawStockList.append(rawStock1, rawStock2, rawStock3, rawStock4, rawStock5, rawStock6, rawStock7, rawStock8)

def showStockInfo(RawStockList, index):
    RawStockList[index].info()

def showStockData(RawStockList, index):
    RawStockList[index].head()
    RawStockList[index].describe()


def calculateMA(RawStockList, index):
    for ma in maDay:
        columnName = "MA for %s  days" %(str(ma))
        MaResult=RawStockList[index]['Adj Close'].rolling(ma).mean()
        print(columnName, MaResult)
        RawStockList[index].insert(6, columnName, RawStockList[index]['Adj Close'].rolling(ma).mean(), True )

def calculateDailyChange(RawStockList, index):
    RawStockList[index].insert(11, 'Daily Return', RawStockList[index]['Adj Close'].pct_change(), True )





#output Raw Stock1 data 
rawStock1.head()
rawStock1.describe()
rawStock2.describe()
rawStock2.head()
rawStock3.describe()
rawStock3.head() 
rawStock4.describe()
rawStock4.head() 
rawStock5.describe()
rawStock5.head() 
rawStock6.describe()
rawStock6.head() 
rawStock7.describe()
rawStock7.head() 
rawStock8.describe()
rawStock8.head()


#calucalte  stock Moving average price
for ma in maDay:
    columnName = "MA for %s  days" %(str(ma))
    MaResult=rawStock1['Adj Close'].rolling(ma).mean()
    print(columnName, MaResult)
    rawStock1.insert(6, columnName, rawStock1['Adj Close'].rolling(ma).mean(), True )
    
for ma in maDay:
    columnName = "MA for %s  days" %(str(ma))
    MaResult=rawStock2['Adj Close'].rolling(ma).mean()
    print(columnName, MaResult)
    rawStock2.insert(6, columnName, rawStock2['Adj Close'].rolling(ma).mean(), True )

for ma in maDay:
    columnName = "MA for %s  days" %(str(ma))
    MaResult=rawStock3['Adj Close'].rolling(ma).mean()
    print(columnName, MaResult)
    rawStock3.insert(6, columnName, rawStock3['Adj Close'].rolling(ma).mean(), True )
    
    
for ma in maDay:
    columnName = "MA for %s  days" %(str(ma))
    MaResult=rawStock4['Adj Close'].rolling(ma).mean()
    print(columnName, MaResult)
    rawStock4.insert(6, columnName, rawStock4['Adj Close'].rolling(ma).mean(), True )
    
for ma in maDay:
    columnName = "MA for %s  days" %(str(ma))
    MaResult=rawStock5['Adj Close'].rolling(ma).mean()
    print(columnName, MaResult)
    rawStock5.insert(6, columnName, rawStock5['Adj Close'].rolling(ma).mean(), True )
    
for ma in maDay:
    columnName = "MA for %s  days" %(str(ma))
    MaResult=rawStock6['Adj Close'].rolling(ma).mean()
    print(columnName, MaResult)
    rawStock6.insert(6, columnName, rawStock6['Adj Close'].rolling(ma).mean(), True )
    
for ma in maDay:
    columnName = "MA for %s  days" %(str(ma))
    MaResult=rawStock7['Adj Close'].rolling(ma).mean()
    print(columnName, MaResult)
    rawStock7.insert(6, columnName, rawStock7['Adj Close'].rolling(ma).mean(), True )
    

for ma in maDay:
    columnName = "MA for %s  days" %(str(ma))
    MaResult=rawStock8['Adj Close'].rolling(ma).mean()
    print(columnName, MaResult)
    rawStock8.insert(6, columnName, rawStock8['Adj Close'].rolling(ma).mean(), True )


#output raw Stock info data 
rawStock1.info 
rawStock2.info 
rawStock3.info 
rawStock4.info 
rawStock5.info 
rawStock6.info 
rawStock7.info
rawStock8.info

# We'll use pct_change to find the percent change for each day
rawStock1.insert(11, 'Daily Return', rawStock1['Adj Close'].pct_change(), True )
rawStock2.insert(11, 'Daily Return', rawStock2['Adj Close'].pct_change(), True )
rawStock3.insert(11, 'Daily Return', rawStock3['Adj Close'].pct_change(), True )
rawStock4.insert(11, 'Daily Return', rawStock4['Adj Close'].pct_change(), True )
rawStock5.insert(11, 'Daily Return', rawStock5['Adj Close'].pct_change(), True )
rawStock6.insert(11, 'Daily Return', rawStock6['Adj Close'].pct_change(), True )
rawStock7.insert(11, 'Daily Return', rawStock7['Adj Close'].pct_change(), True )
rawStock8.insert(11, 'Daily Return', rawStock8['Adj Close'].pct_change(), True )


# Plot data
plt.figure(figsize= (20, 10))
plt.plot(range(rawStock1.shape[0]),(rawStock1['Low']+rawStock1['High'])/2.0, label='Average Price')
plt.plot(range(rawStock1.shape[0]),(rawStock1['Adj Close']), label='Adj')
plt.plot(range(rawStock1.shape[0]),(rawStock1['Open']), label='Open')
plt.plot(range(rawStock1.shape[0]),(rawStock1['Close']), label='Close')
plt.xticks(range(0,rawStock1.shape[0],50),rawStock1.index[::50], rotation=60)
plt.title('Stock1: Price')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.legend(loc='best')
plt.show()


plt.figure(figsize= (20, 10))
plt.plot(range(rawStock1.shape[0]),(rawStock1['Volume']), label='Volume')
plt.xticks(range(0,rawStock1.shape[0],50),rawStock1.index[::50], rotation=60)
plt.title('Stock1: Volume History')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Volume',fontsize=18)
plt.legend(loc='best')
plt.show()

plt.figure(figsize= (20, 12))
plt.plot(range(rawStock1.shape[0]),(rawStock1['Low']+rawStock1['High'])/2.0, label='Average Price')
plt.plot(range(rawStock1.shape[0]),(rawStock1['Adj Close']), label='Adj')
plt.plot(range(rawStock1.shape[0]),(rawStock1['MA for 200  days']), label='MA for 200  days')
plt.plot(range(rawStock1.shape[0]),(rawStock1['MA for 50  days']), label='MA for 50  days')
plt.plot(range(rawStock1.shape[0]),(rawStock1['MA for 20  days']), label='MA for 20  days')
plt.plot(range(rawStock1.shape[0]),(rawStock1['MA for 10  days']), label='MA for 10  days')
plt.xticks(range(0,rawStock1.shape[0],100),rawStock1.index[::100], rotation=60)
plt.title('Stock1: Moving  Analysis')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.legend(loc='best')
plt.show()

# Then we'll plot the daily return percentage
rawStock1['Daily Return'].plot(figsize=(20,10), legend=True,linestyle='--',marker='o')

plt.figure(figsize= (20, 10))
plt.plot(range(rawStock2.shape[0]),(rawStock2['Low']+rawStock2['High'])/2.0, label='Average Price')
plt.plot(range(rawStock2.shape[0]),(rawStock2['Adj Close']), label='Adj')
plt.plot(range(rawStock2.shape[0]),(rawStock2['Open']), label='Open')
plt.plot(range(rawStock2.shape[0]),(rawStock2['Close']), label='Close')
plt.xticks(range(0,rawStock2.shape[0],100),rawStock2.index[::100], rotation=60)
plt.title('Stock2: Price')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.legend(loc='best')
plt.show()

plt.figure(figsize= (20, 10))
plt.plot(range(rawStock2.shape[0]),(rawStock2['Volume']), label='Volume')
plt.xticks(range(0,rawStock2.shape[0],50),rawStock2.index[::50], rotation=60)
plt.title('Stock2: Volume History')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Volume',fontsize=18)
plt.legend(loc='best')
plt.show()

plt.figure(figsize= (20, 12))
plt.plot(range(rawStock2.shape[0]),(rawStock2['Low']+rawStock2['High'])/2.0, label='Average Price')
plt.plot(range(rawStock2.shape[0]),(rawStock2['Adj Close']), label='Adj')
plt.plot(range(rawStock2.shape[0]),(rawStock2['MA for 200  days']), label='MA for 200  days')
plt.plot(range(rawStock2.shape[0]),(rawStock2['MA for 50  days']), label='MA for 50  days')
plt.plot(range(rawStock2.shape[0]),(rawStock2['MA for 20  days']), label='MA for 20  days')
plt.plot(range(rawStock2.shape[0]),(rawStock2['MA for 10  days']), label='MA for 10  days')
plt.xticks(range(0,rawStock2.shape[0],100),rawStock2.index[::100], rotation=60)
plt.title('Stock2: Moving  Analysis')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.legend(loc='best')
plt.show()

# Then we'll plot the daily return percentage
rawStock2['Daily Return'].plot(figsize=(20,10), legend=True,linestyle='--',marker='o')

plt.figure(figsize= (20, 10))
plt.plot(range(rawStock3.shape[0]),(rawStock3['Low']+rawStock3['High'])/2.0, label='Average Price')
plt.plot(range(rawStock3.shape[0]),(rawStock3['Adj Close']), label='Adj')
plt.plot(range(rawStock3.shape[0]),(rawStock3['Open']), label='Open')
plt.plot(range(rawStock3.shape[0]),(rawStock3['Close']), label='Close')
plt.xticks(range(0,rawStock3.shape[0],50),rawStock3.index[::50], rotation=60)
plt.title('Stock3: Price')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.legend(loc='best')
plt.show()

plt.figure(figsize= (20, 10))
plt.plot(range(rawStock3.shape[0]),(rawStock3['Volume']), label='Volume')
plt.xticks(range(0,rawStock3.shape[0],50),rawStock3.index[::50], rotation=60)
plt.title('Stock3: Volume History')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Volume',fontsize=18)
plt.legend(loc='best')
plt.show()

plt.figure(figsize= (20, 12))
plt.plot(range(rawStock3.shape[0]),(rawStock3['Low']+rawStock3['High'])/2.0, label='Average Price')
plt.plot(range(rawStock3.shape[0]),(rawStock3['Adj Close']), label='Adj')
plt.plot(range(rawStock3.shape[0]),(rawStock3['MA for 200  days']), label='MA for 200  days')
plt.plot(range(rawStock3.shape[0]),(rawStock3['MA for 50  days']), label='MA for 50  days')
plt.plot(range(rawStock3.shape[0]),(rawStock3['MA for 20  days']), label='MA for 20  days')
plt.plot(range(rawStock3.shape[0]),(rawStock3['MA for 10  days']), label='MA for 10  days')
plt.xticks(range(0,rawStock3.shape[0],100),rawStock3.index[::100], rotation=60)
plt.title('Stock3: Moving  Analysis')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.legend(loc='best')
plt.show()

# Then we'll plot the daily return percentage
rawStock3['Daily Return'].plot(figsize=(20,10), legend=True,linestyle='--',marker='o')


# Plot data
plt.figure(figsize= (20, 10))
plt.plot(range(rawStock4.shape[0]),(rawStock4['Low']+rawStock4['High'])/2.0, label='Average Price')
plt.plot(range(rawStock4.shape[0]),(rawStock4['Adj Close']), label='Adj')
plt.plot(range(rawStock4.shape[0]),(rawStock4['Open']), label='Open')
plt.plot(range(rawStock4.shape[0]),(rawStock4['Close']), label='Close')
plt.xticks(range(0,rawStock4.shape[0],50),rawStock4.index[::50], rotation=60)
plt.title('Stock4: Price')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.legend(loc='best')
plt.show()


plt.figure(figsize= (20, 10))
plt.plot(range(rawStock4.shape[0]),(rawStock4['Volume']), label='Volume')
plt.xticks(range(0,rawStock4.shape[0],50),rawStock4.index[::50], rotation=60)
plt.title('Stock4: Volume History')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Volume',fontsize=18)
plt.legend(loc='best')
plt.show()

plt.figure(figsize= (20, 12))
plt.plot(range(rawStock4.shape[0]),(rawStock4['Low']+rawStock4['High'])/2.0, label='Average Price')
plt.plot(range(rawStock4.shape[0]),(rawStock4['Adj Close']), label='Adj')
plt.plot(range(rawStock4.shape[0]),(rawStock4['MA for 200  days']), label='MA for 200  days')
plt.plot(range(rawStock4.shape[0]),(rawStock4['MA for 50  days']), label='MA for 50  days')
plt.plot(range(rawStock4.shape[0]),(rawStock4['MA for 20  days']), label='MA for 20  days')
plt.plot(range(rawStock4.shape[0]),(rawStock4['MA for 10  days']), label='MA for 10  days')
plt.xticks(range(0,rawStock4.shape[0],100),rawStock4.index[::100], rotation=60)
plt.title('Stock4: Moving  Analysis')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.legend(loc='best')
plt.show()

# Then we'll plot the daily return percentage
rawStock4['Daily Return'].plot(figsize=(20,10), legend=True,linestyle='--',marker='o')


# Plot data
plt.figure(figsize= (20, 10))
plt.plot(range(rawStock5.shape[0]),(rawStock5['Low']+rawStock5['High'])/2.0, label='Average Price')
plt.plot(range(rawStock5.shape[0]),(rawStock5['Adj Close']), label='Adj')
plt.plot(range(rawStock5.shape[0]),(rawStock5['Open']), label='Open')
plt.plot(range(rawStock5.shape[0]),(rawStock5['Close']), label='Close')
plt.xticks(range(0,rawStock5.shape[0],50),rawStock5.index[::50], rotation=60)
plt.title('Stock5: Price')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.legend(loc='best')
plt.show()


plt.figure(figsize= (20, 10))
plt.plot(range(rawStock5.shape[0]),(rawStock5['Volume']), label='Volume')
plt.xticks(range(0,rawStock5.shape[0],50),rawStock5.index[::50], rotation=60)
plt.title('Stock5: Volume History')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Volume',fontsize=18)
plt.legend(loc='best')
plt.show()


plt.figure(figsize= (20, 12))
plt.plot(range(rawStock5.shape[0]),(rawStock5['Low']+rawStock5['High'])/2.0, label='Average Price')
plt.plot(range(rawStock5.shape[0]),(rawStock5['Adj Close']), label='Adj')
plt.plot(range(rawStock5.shape[0]),(rawStock5['MA for 200  days']), label='MA for 200  days')
plt.plot(range(rawStock5.shape[0]),(rawStock5['MA for 50  days']), label='MA for 50  days')
plt.plot(range(rawStock5.shape[0]),(rawStock5['MA for 20  days']), label='MA for 20  days')
plt.plot(range(rawStock5.shape[0]),(rawStock5['MA for 10  days']), label='MA for 10  days')
plt.xticks(range(0,rawStock5.shape[0],100),rawStock5.index[::100], rotation=60)
plt.title('Stock5: Moving  Analysis')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.legend(loc='best')
plt.show()

# Then we'll plot the daily return percentage
rawStock5['Daily Return'].plot(figsize=(20,10), legend=True,linestyle='--',marker='o')


# Plot data
plt.figure(figsize= (20, 10))
plt.plot(range(rawStock6.shape[0]),(rawStock6['Low']+rawStock6['High'])/2.0, label='Average Price')
plt.plot(range(rawStock6.shape[0]),(rawStock6['Adj Close']), label='Adj')
plt.plot(range(rawStock6.shape[0]),(rawStock6['Open']), label='Open')
plt.plot(range(rawStock6.shape[0]),(rawStock6['Close']), label='Close')
plt.xticks(range(0,rawStock6.shape[0],50),rawStock6.index[::50], rotation=60)
plt.title('Stock6: Price')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.legend(loc='best')
plt.show()


# Plot data
plt.figure(figsize= (20, 10))
plt.plot(range(rawStock7.shape[0]),(rawStock7['Low']+rawStock7['High'])/2.0, label='Average Price')
plt.plot(range(rawStock7.shape[0]),(rawStock7['Adj Close']), label='Adj')
plt.plot(range(rawStock7.shape[0]),(rawStock7['Open']), label='Open')
plt.plot(range(rawStock7.shape[0]),(rawStock7['Close']), label='Close')
plt.xticks(range(0,rawStock7.shape[0],50),rawStock7.index[::50], rotation=45)
plt.title('Stock7: Price')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.legend(loc='best')
plt.show()

plt.figure(figsize= (20, 10))
plt.plot(range(rawStock7.shape[0]),(rawStock7['Volume']), label='Volume')
plt.xticks(range(0,rawStock7.shape[0],50),rawStock7.index[::50], rotation=60)
plt.title('Stock7: Volume History')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Volume',fontsize=18)
plt.legend(loc='best')
plt.show()


plt.figure(figsize= (20, 12))
plt.plot(range(rawStock7.shape[0]),(rawStock7['Low']+rawStock7['High'])/2.0, label='Average Price')
plt.plot(range(rawStock7.shape[0]),(rawStock7['Adj Close']), label='Adj')
plt.plot(range(rawStock7.shape[0]),(rawStock7['MA for 200  days']), label='MA for 200  days')
plt.plot(range(rawStock7.shape[0]),(rawStock7['MA for 50  days']), label='MA for 50  days')
plt.plot(range(rawStock7.shape[0]),(rawStock7['MA for 20  days']), label='MA for 20  days')
plt.plot(range(rawStock7.shape[0]),(rawStock7['MA for 10  days']), label='MA for 10  days')
plt.xticks(range(0,rawStock7.shape[0],100),rawStock7.index[::100], rotation=60)
plt.title('Stock5: Moving  Analysis')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.legend(loc='best')
plt.show()

# Then we'll plot the daily return percentage
rawStock3['Daily Return'].plot(figsize=(20,10), legend=True,linestyle='--',marker='o')


# Plot data
plt.figure(figsize= (20, 10))
plt.plot(range(rawStock8.shape[0]),(rawStock8['Low']+rawStock8['High'])/2.0, label='Average Price')
plt.plot(range(rawStock8.shape[0]),(rawStock8['Adj Close']), label='Adj')
plt.plot(range(rawStock8.shape[0]),(rawStock8['Open']), label='Open')
plt.plot(range(rawStock8.shape[0]),(rawStock8['Close']), label='Close')
plt.xticks(range(0,rawStock8.shape[0],50),rawStock8.index[::50], rotation=45)
plt.title('Stock8: Price')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.legend(loc='best')
plt.show()

plt.figure(figsize= (20, 10))
plt.plot(range(rawStock8.shape[0]),(rawStock8['Volume']), label='Volume')
plt.xticks(range(0,rawStock8.shape[0],50),rawStock8.index[::50], rotation=60)
plt.title('Stock8: Volume History')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Volume',fontsize=18)
plt.legend(loc='best')
plt.show()

plt.figure(figsize= (20, 12))
plt.plot(range(rawStock8.shape[0]),(rawStock8['Low']+rawStock8['High'])/2.0, label='Average Price')
plt.plot(range(rawStock8.shape[0]),(rawStock8['Adj Close']), label='Adj')
plt.plot(range(rawStock8.shape[0]),(rawStock8['MA for 200  days']), label='MA for 200  days')
plt.plot(range(rawStock8.shape[0]),(rawStock8['MA for 50  days']), label='MA for 50  days')
plt.plot(range(rawStock8.shape[0]),(rawStock8['MA for 20  days']), label='MA for 20  days')
plt.plot(range(rawStock8.shape[0]),(rawStock8['MA for 10  days']), label='MA for 10  days')
plt.xticks(range(0,rawStock8.shape[0],100),rawStock8.index[::100], rotation=60)
plt.title('Stock8: Moving  Analysis')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.legend(loc='best')
plt.show()

# Then we'll plot the daily return percentage
rawStock8['Daily Return'].plot(figsize=(20,10), legend=True,linestyle='--',marker='o')


def plotPrice(RawStockList, index):
    plt.figure(figsize= (20, 10))
    plt.plot(range(RawStockList[index].shape[0]),(RawStockList[index]['Low']+RawStockList[index]['High'])/2.0, label='Average Price')
    plt.plot(range(RawStockList[index]),(RawStockList[index]['Adj Close']), label='Adj')
    plt.plot(range(RawStockList[index]),(RawStockList[index]['Open']), label='Open')
    plt.plot(range(RawStockList[index]),(RawStockList[index]['Close']), label='Close')
    plt.xticks(range(0,RawStockList[index].shape[0],50),RawStockList[index].index[::50], rotation=45)
    plt.title('Stock %d : Price' %index)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Mid Price',fontsize=18)
    plt.legend(loc='best')
    plt.show()
    
def plotVolume(RawStockList, index):
    plt.figure(figsize= (20, 10))
    plt.plot(range(RawStockList[index].shape[0]),(RawStockList[index]['Volume']), label='Volume')
    plt.xticks(range(0,RawStockList[index].shape[0],50),RawStockList[index].index[::50], rotation=60)
    plt.title('Stock %d : Volume History' %index)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Volume',fontsize=18)
    plt.legend(loc='best')
    plt.show()
    

def plotDailyChange(RawStockList, index):
    # Then we'll plot the daily return percentage
    RawStockList[index]['Daily Return'].plot(figsize=(20,10), legend=True,linestyle='--',marker='o')
    plt.show()


for index in range(0, len(RawStockList)):
    plotPrice(RawStockList, index)
    plotVolume(RawStockList, index)
    plotDailyChange(RawStockList, index)



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

# First calculate the mid prices from the highest and lowest
def CalculateNorm(RawStockList, index):
    high_prices = RawStockList[index].loc[:,'High'].as_matrix()
    low_prices = RawStockList[index].loc[:,'Low'].as_matrix()
    mid_prices = (high_prices+low_prices)/2.0

    return high_prices, low_prices,  mid_prices


trainHigPrices , trainLowPrice,  trainMidPrice = CalculateNorm(RawStockList, 0)

train_data = trainMidPrice[:300]
test_data = trainMidPrice[300:]