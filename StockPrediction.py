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

import sklearn
import sklearn.preprocessing

#from sklearn.metrics import accuracy_score, confusion_matrix,  classification_report


import matplotlib.pyplot as plt

import datetime

import memory_profiler 

import psutil

EnableGPU =False

train_split =0.8

# define DataFrame column index
OpenIndex =  'Open'
CloseIndex = 'Close'
HighIndex = 'High'
LowIndex = 'Low'
AdjCloseIndex = 'Adj Close'
VolumeIndex = 'Volume'

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

#define Key for dictionary
RawStock1Key = 'rawStock1'
RawStock2Key = 'rawStock2'
RawStock3Key = 'rawStock3'
RawStock4Key = 'rawStock4'
RawStock5Key = 'rawStock5'
RawStock6Key = 'rawStock6'
RawStock7Key = 'rawStock7'
RawStock8Key = 'rawStock8'

PreTrainStock1Key  ='preTrainStock1'
PreTrainStock2Key  ='preTrainStock2'
PreTrainStock3Key  ='preTrainStock3'
PreTrainStock4Key  ='preTrainStock4'
PreTrainStock5Key  ='preTrainStock5'
PreTrainStock6Key  ='preTrainStock6'
PreTrainStock7Key  ='preTrainStock7'
PreTrainStock8Key  ='preTrainStock8'



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
# create RowStock Dictionary 
RawStockList = {}

RawStockList[RawStock1Key] = rawStock1
RawStockList[RawStock2Key] = rawStock2
RawStockList[RawStock3Key] = rawStock3
RawStockList[RawStock4Key] = rawStock4
RawStockList[RawStock5Key] = rawStock5
RawStockList[RawStock6Key] = rawStock6
RawStockList[RawStock7Key] = rawStock7
RawStockList[RawStock8Key] = rawStock8

# create preTrainStock Dictionary 
PreTrainStockList = {}

PreTrainStockList[PreTrainStock1Key] = preTrainStock1
PreTrainStockList[PreTrainStock2Key] = preTrainStock2
PreTrainStockList[PreTrainStock3Key] = preTrainStock3
PreTrainStockList[PreTrainStock4Key] = preTrainStock4
PreTrainStockList[PreTrainStock5Key] = preTrainStock5
PreTrainStockList[PreTrainStock6Key] = preTrainStock6
PreTrainStockList[PreTrainStock7Key] = preTrainStock7
PreTrainStockList[PreTrainStock8Key] = preTrainStock8


def showStockInfo(Stock):
    print(Stock.info())
    print("Stock Matrix Size: " , Stock.shape)
    print("Stock Martix Row :" ,Stock.shape[0])
    print("Stock Martix Column :" , Stock.shape[1])
    print("Stock Martix Total data :" , Stock.size)


def showStockHead(Stock):
    print(Stock.head())
    
def showStockDescribe(Stock):
    print(Stock.describe())    

def showStockTail(Stock):
    print(Stock.tail())

def checkStockNullData(Stock):
    print(Stock.isna().sum())

def calculateMA(Stock):
    for ma in maDay:
        columnName = "MA for %s  days" %(str(ma))
        MaResult=Stock['Adj Close'].rolling(ma).mean()
        print(columnName, MaResult)
        Stock.insert(6, columnName, Stock['Adj Close'].rolling(ma).mean(), True )

def calculateDailyChange(Stock):
    Stock.insert(11, 'Daily Return', Stock['Adj Close'].pct_change(), True )
    

# First calculate the mid prices from the highest and lowest
def CalculateAvergePrice(Stock):
    high_prices = Stock.loc[:,'High'].to_numpy()
    low_prices = Stock.loc[:,'Low'].to_numpy()
    mid_prices = (high_prices+low_prices)/2.0

    return high_prices, low_prices,  mid_prices

def augFeatures(Stock):
    Stock["year"] = pd.DatetimeIndex(Stock.iloc[:,0]).year
    Stock["month"] = pd.DatetimeIndex(Stock.iloc[:,0]).month
    Stock["date"] = pd.DatetimeIndex(Stock.iloc[:,0]).day
    Stock["day"] =  pd.DatetimeIndex(Stock.iloc[:,0]).dayofweek
    

def CalculateNormalize(Stock):
    StockTemp = Stock.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x))) # Z
    #StockTemp = Stock.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))) # min-max
    return StockTemp

def ScaleDataNorm(Stock):
    StockTemp = Stock.copy()
    min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range =(-1,1))
    StockTemp['Open'] = min_max_scaler.fit_transform(StockTemp['Open'].values.reshape(-1, 1))
    StockTemp['High'] = min_max_scaler.fit_transform(StockTemp['High'].values.reshape(-1, 1))
    StockTemp['Low'] = min_max_scaler.fit_transform(StockTemp['Low'].values.reshape(-1, 1))
    StockTemp['Close'] = min_max_scaler.fit_transform(StockTemp['Close'].values.reshape(-1, 1))
    StockTemp['Adj Close'] = min_max_scaler.fit_transform(StockTemp['Adj Close'].values.reshape(-1, 1))
    #StockTemp['Volume']  = min_max_scaler.fit_transform(StockTemp['Volume'].values.reshape(-1, 1))
    return StockTemp

def ScaleColumnData(Col, Min, Max, Fit=False):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range =(Min,Max))
    if(Fit):
        OutData = min_max_scaler.fit_transform(Col)
    else:
        OutData = min_max_scaler.transform(Col)
    return OutData , min_max_scaler
        
def showAllStockInfo(StockList):
    for stockTempKey, stockTempValue in StockList.items():
        print("\n\rShow Stock New Info :", stockTempKey)
        showStockInfo(stockTempValue)

def showAllStockHead(StockList):
    for stockTempKey, stockTempValue in StockList.items():
        print("\n\rShow Stock Head :", stockTempKey)
        showStockHead(stockTempValue)

def showAllStockTail(StockList):
    for stockTempKey, stockTempValue in StockList.items():
        print("\n\rShow Stock Tail :", stockTempKey)
        showStockTail(stockTempValue)

def showAllStockNullData(StockList):
     for stockTempKey, stockTempValue in StockList.items():
        print("\n\rShow Stock Null Data :", stockTempKey)
        checkStockNullData(stockTempValue)
    

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


# start main program
resultEpoch = []
resultLoss = []    



showAllStockInfo(RawStockList)
showAllStockHead(RawStockList)
showAllStockTail(RawStockList)
showAllStockNullData(RawStockList)


for stockTempKey, stockTempValue in RawStockList.items(): 
    print("\n\rCalculate MA for Stock :", stockTempKey)
    calculateMA(stockTempValue)
    print("\n\rCalculate Daily Change for Stock :", stockTempKey)
    calculateDailyChange(stockTempValue)
    print("\n\rCalculate High/Low/Average Price :", stockTempKey)
    trainHigPrices , trainLowPrice,  trainMidPrice = CalculateAvergePrice(stockTempValue)
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
showAllStockHead(PreTrainStockList)
showAllStockTail(PreTrainStockList)

print("print Raw Stock1 :", RawStockList[RawStock1Key])
print("print Raw Stock1 index column:", RawStockList[RawStock1Key].iloc[:,0])

#augFeatures(RawStockList[RawStock1Key])
showStockInfo(RawStockList[RawStock1Key])

'''
# Data normalize scaler
normalizeStock1 = CalculateNormalize(RawStockList[RawStock1Key])
showStockTail(normalizeStock1)
print("print normalize Stock : ", normalizeStock1.iloc[3])
print("print RawStock1 : ", RawStockList[RawStock1Key].iloc[3])

normalizeStock11 = ScaleDataNorm(RawStockList[RawStock1Key])
showStockHead(normalizeStock11)
print("print normalize Stock : ", normalizeStock11.iloc[3])
'''

#Splitting data into training set and a test set 
num_data = RawStockList[RawStock1Key].shape[0]
print("Number of data size of stock1 : ", num_data)
num_train = ((int)(train_split * num_data))
print("Number of train data of stock1 : ", num_train)
train_data = RawStockList[RawStock1Key][AdjCloseIndex][: num_train]
test_data =  RawStockList[RawStock1Key][AdjCloseIndex][num_train:]
print("Train data  : ", train_data , ' size :', train_data.shape[0] , 'Shape :',train_data.shape )
print("Test data  : ", test_data, ' size :', test_data.shape[0] , 'Shape :',test_data.shape)
#reshape 1D data into 2D metrix data before feed LSTM model 
print("train_data  before reshape: ", train_data, ' size :', train_data.shape[0] , 'Shape :',train_data.shape )
train_data = train_data.values.reshape(-1,1)
test_data  = test_data.values.reshape(-1,1)
print("train_data reshape -1 and 1 : ", train_data, ' size :', train_data.shape[0], 'Shape :',train_data.shape )

#Data normalize scaler
train_data , trainScalar = ScaleColumnData(train_data, 0, 1, True)
test_data , testScalar = ScaleColumnData(test_data, 0, 1, True)
print("train_data  After Sacle: ", train_data, ' size :', train_data.shape[0] , 'Shape :',train_data.shape )
print("test_data  After Sacle: ", test_data, ' size :', test_data.shape[0] , 'Shape :',test_data.shape)


# Globals

INPUT_SIZE = 50
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1

# Hyper parameters

learning_rate = 0.01# 0.001
num_epochs = 1000

# Creating a data structure with 60 timesteps and 1 output
# x_train for input sequence
# y_train for target sequence
X_train = []
y_train = []
for i in range(INPUT_SIZE, train_data.shape[0]):
    X_train.append(train_data[i-INPUT_SIZE:i, 0])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
print("X_Train shape: ", X_train.shape)
#print(X_train)
print("Y_Train shape: ", y_train.shape)
#print(y_train)

# Reshaping 3 dimension data
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
print("X_train Shape after reshape: ", X_train.shape)

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
    
def LTSMStockTrain(hidden_state, model):
    for epoch in range(num_epochs):
        inputs = Variable(torch.from_numpy(X_train).float())
        labels = Variable(torch.from_numpy(y_train).float())
        
        output = model(inputs) 
        loss = criterion(output.view(-1), labels)
        optimiser.zero_grad()
        loss.backward(retain_graph=True)                     # back propagation
        optimiser.step()                                     # update the parameters
        if epoch % 50 == 0:
            print('epoch {}, loss {}'.format(epoch,loss.item()))
        resultEpoch.append(epoch)
        resultLoss.append(loss.item())


model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
 # loss function
criterion = nn.MSELoss() 

hidden_state = None
LTSMStockTrain(hidden_state, model)



#recovery origin train test data 
print('train data shape :', train_data.shape, type(train_data))
print('test data shape :', test_data.shape, type(test_data))
#inverse scaler
originTrain = trainScalar.inverse_transform(train_data)
originTest  = testScalar.inverse_transform(test_data)
origin_data = np.concatenate((originTrain, originTest), axis=0)

#Test model 
testInputs = origin_data.reshape(-1, 1)
testInputs = trainScalar.transform(testInputs)
X_test = []
for i in range(INPUT_SIZE, 80):
    X_test.append(testInputs[i-INPUT_SIZE:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

X_train_X_test = np.concatenate((X_train, X_test),axis=0)
hidden_state = None
test_inputs = Variable(torch.from_numpy(X_train_X_test).float())
predicted_stock_price = model(test_inputs)
predicted_stock_price = np.reshape(predicted_stock_price.detach().numpy(), (test_inputs.shape[0], 1))
predicted_stock_price = trainScalar.inverse_transform(predicted_stock_price)



plt.figure(figsize=(10,6))
plt.plot(resultEpoch, resultLoss)
plt.xlabel('Step (Training)')
plt.ylabel('Loss (%)')
plt.title('Loss Vs Number of Step')
plt.show()



print('origin data shape :', origin_data.shape)
plt.figure(figsize= (10,6))
plt.plot(origin_data, color = 'blue' ,label = 'Origin Price')
plt.plot(predicted_stock_price, color = 'red' ,label = 'Predict Price')
plt.xlabel('Date Time')
plt.ylabel('Price')
plt.title('Predict Price Result')
plt.legend()
plt.show()




