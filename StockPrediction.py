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

#import memory_profiler 

import psutil

EnableGPU =True

train_split = 0.8 #0.99

# CPU to GPU
if torch.cuda.is_available() and EnableGPU:
  #  tensor_cpu.cuda()
    torch.cuda.empty_cache()
    print("GPU is available")
    device = torch.device("cuda:0") # Uncomment this to run on GPU
    print("GPU Name: ", torch.cuda.get_device_name())
    
else: 
    print('No GPU')
    
print("PyTorch Version: ", torch.__version__)

# define DataFrame column index
OpenIndex =  'Open'
CloseIndex = 'Close'
HighIndex = 'High'
LowIndex = 'Low'
AdjCloseIndex = 'Adj Close'
VolumeIndex = 'Volume'

# Declare Train stock  path 
PreTrainStock1 = 'stock_dataset_v3/preprocess_price_train/1_p_price_train.txt'
PreTrainStock2 = 'stock_dataset_v3/preprocess_price_train/2_p_price_train.txt'
PreTrainStock3 = 'stock_dataset_v3/preprocess_price_train/3_p_price_train.txt'
PreTrainStock4 = 'stock_dataset_v3/preprocess_price_train/4_p_price_train.txt'
PreTrainStock5 = 'stock_dataset_v3/preprocess_price_train/5_p_price_train.txt'
PreTrainStock6 = 'stock_dataset_v3/preprocess_price_train/6_p_price_train.txt'
PreTrainStock7 = 'stock_dataset_v3/preprocess_price_train/7_p_price_train.txt'
PreTrainStock8 = 'stock_dataset_v3/preprocess_price_train/8_p_price_train.txt'

# Declare Raw stock  path 
RawStock1 = 'stock_dataset_v3/raw_price_train/1_r_price_train.csv'
RawStock2 = 'stock_dataset_v3/raw_price_train/2_r_price_train.csv'
RawStock3 = 'stock_dataset_v3/raw_price_train/3_r_price_train.csv'
RawStock4 = 'stock_dataset_v3/raw_price_train/4_r_price_train.csv'
RawStock5 = 'stock_dataset_v3/raw_price_train/5_r_price_train.csv'
RawStock6 = 'stock_dataset_v3/raw_price_train/6_r_price_train.csv'
RawStock7 = 'stock_dataset_v3/raw_price_train/7_r_price_train.csv'
RawStock8 = 'stock_dataset_v3/raw_price_train/8_r_price_train.csv'

#Declare Tweet Directory 
TrainTweetDirectory = 'stock_dataset_v3/tweet_train/'

TestTweetDirectory = 'stock_dataset_v3/tweet_test/'



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


predictFutureDate = []

predictFutureDate.append('2015-12-21')
predictFutureDate.append('2015-12-22')
predictFutureDate.append('2015-12-23')
predictFutureDate.append('2015-12-24')
predictFutureDate.append('2015-12-28')
predictFutureDate.append('2015-12-29')
predictFutureDate.append('2015-12-30')

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
    #plt.plot(range(Stock.shape[0]),(Stock['Low']+Stock['High'])/2.0, label='Average Price')
    plt.plot(range(Stock.shape[0]),(Stock['Adj Close']), label='Adj')
    #plt.plot(range(Stock.shape[0]),(Stock['Open']), label='Open')
    #plt.plot(range(Stock.shape[0]),(Stock['Close']), label='Close')
    plt.xticks(range(0,Stock.shape[0],50),Stock.index[::50], rotation=45)
    plt.title('Stock %s : Price' %name)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Adj Close Price',fontsize=18)
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

def plotLossResult(Epoch, Loss):
    plt.figure(figsize=(12,8))
    plt.plot(Epoch, Loss)
    plt.xlabel('Step (Training)')
    plt.ylabel('Loss (%)')
    plt.title('Loss Vs Number of Step')
    plt.show()

def plotPredictByHistory(realPrice, predictPrice, Stock):
    plt.figure(figsize= (12,8))
    plt.plot(realPrice, color = 'blue' ,label = 'Real Price')
    plt.plot(predictPrice, color = 'red' ,label = 'Predict Price')
    plt.xticks(range(0, Stock.shape[0],50), Stock.index[60::50], rotation=45)
    plt.xlabel('Date Time')
    plt.ylabel('Price')
    plt.title('Predict Whole Price Result')
    plt.legend()
    plt.show()

def plotPredictFuturePrice(predictOut):
    plt.figure(figsize= (12,8))
    #plt.plot(origin_data, color = 'blue' ,label = 'Origin Price')
    #plt.plot(real_stock_price_all[-INPUT_SIZE:], color = 'blue' ,label = 'Real Price')
    plt.plot(predictOut[1:], color = 'red' ,label = 'Predict N Future Days Price')
    #plt.xticks(range(0, TestStock.shape[0],20),TestStock.index[750+60::20], rotation=45)
    plt.xlabel('Next Time (days)')
    plt.ylabel('Predict Price')
    #plt.yticks(np.arange(0, max(predictOut), 20)) 
    plt.xticks(range(0, TestPredictDay),predictFutureDate ) 
    plt.title('Predict N Day Result Zoom ')
    plt.legend()
    plt.show()



# start main program


'''

#calculate MA for each Stock
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
'''


#plotting each stock
for stockTempKey, stockTempValue in RawStockList.items():
    print("\n\rPlot Price :", stockTempKey)
    plotPrice(stockTempValue, stockTempKey)
    plotVolume(stockTempValue, stockTempKey)
    #plotDailyChange(stockTempValue, stockTempKey)


# Globals

INPUT_SIZE = 60 # this depend on 
MaxTestRange = INPUT_SIZE + 6
HIDDEN_SIZE = 100
NUM_LAYERS = 3
OUTPUT_SIZE = 1
TestPredictDay = 7

# Hyper parameters

learning_rate = 0.0005# 0.001
#num_epochs = 250
num_epochs = 50#250

def TrainStockPrepare(TestDict, TestStockKey, TestColumn):
    #Splitting data into training set and a test set 
    TestStock = None
    TestStock = TestDict[TestStockKey]
    #showStockHead(TestStock)
    #print(TestStock)
    #showStockTail(TestStock)
    num_data = TestStock.shape[0]
    print("Number of data size of Test Stock : ", num_data)
    num_train = ((int)(train_split * num_data))
    #num_train = ((int)(num_data-train_split ))
    print("Number of train data of stock1 : ", num_train)


    #Data normalize scaler with reshape 1D data into 2D metrix data before feed LSTM model 
    print('RawStock', TestStock[TestColumn].shape)
    rawStockData =TestStock[TestColumn].values.reshape(-1,1)
    #print(type(rawStockData))

    scaledData , trainScalar = ScaleColumnData(rawStockData, -1, 1, True)
    #print('Scaled Data:', scaledData)
    #split data 
    train_data = None
    test_data = None
    train_data = scaledData[: num_train] 
    test_data =  scaledData[num_train:]
    print("Train data size :", train_data.shape[0] , 'Shape :',train_data.shape )
    print("Test data  size :", test_data.shape[0] , 'Shape :',test_data.shape)

    # Creating a data structure with 60 timesteps and 1 output
    # x_train for input sequence
    # y_train for target sequence
    X_train = []
    y_train = []
    hidden_state = None
    for i in range(INPUT_SIZE, train_data.shape[0]):
        X_train.append(train_data[i-INPUT_SIZE:i, 0])
        #y_train.append(train_data[i, 0])
        y_train.append(train_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    print("X_Train shape: ", X_train.shape)
    print("Y_Train shape: ", y_train.shape)#print(X_train)
    #y_train.reshape(y_train.shape[0], 2)
    #y_train.reshape(((int)(y_train.shape[0]/OUTPUT_SIZE)), OUTPUT_SIZE)
    print("Y_Train shape: ", y_train.shape)
    #print(y_train)

    # Reshaping 3 dimension data
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    print("X_train Shape after reshape: ", X_train.shape)

    return  X_train, y_train, train_data, test_data, TestStock, trainScalar ,hidden_state

class LSTMModel(nn.Module):
    def __init__(self, i_size, h_size, n_layers, o_size):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=i_size,
            hidden_size=h_size,
            num_layers=n_layers
        )
        self.out = nn.Linear(h_size, o_size)

    def forward(self, x, h_state):
        r_out, hidden_state = self.lstm(x, h_state)
        
        hidden_size = hidden_state[-1].size(-1)
        r_out = r_out.view(-1, hidden_size)
        outs = self.out(r_out)

        return outs, hidden_state



    
def LTSMStockTrain(X_train, y_train, hidden_state, model, criterion, optimiser, Epoch, Loss):
    print(y_train.shape)
    hiddenState = hidden_state
    for epoch in range(num_epochs):
        if torch.cuda.is_available() and EnableGPU:
            inputs = Variable(torch.from_numpy(X_train).float().cuda())
            labels = Variable(torch.from_numpy(y_train).float().cuda())
        else:
            inputs = Variable(torch.from_numpy(X_train).float())
            labels = Variable(torch.from_numpy(y_train).float())
        
        output, hiddenState  = model(inputs, hiddenState) 
        loss = criterion(output.view(-1), labels)
        optimiser.zero_grad()
        loss.backward(retain_graph=True)                     # back propagation
        optimiser.step()                                     # update the parameters
        if epoch % 5 == 0:
            print('epoch {}, loss {}'.format(epoch,loss.item()))
        Epoch.append(epoch)
        Loss.append(loss.item())
        
def predictByTrainHistory(X_train, train_data, test_data, trainScalar, fileName):
    X_test = []
    #recovery origin train test data 
    print('train data shape :', train_data.shape, type(train_data))
    print('test data shape :', test_data.shape, type(test_data))
    origin_data = np.concatenate((train_data, test_data), axis=0)
    #inverse scaler
    origin_data = trainScalar.inverse_transform(origin_data)


    #Test model for predict 
    testInputs= origin_data[origin_data.shape[0]- test_data.shape[0] -INPUT_SIZE :]
    testInputs = testInputs.reshape(-1, 1)
    testInputs = trainScalar.transform(testInputs)
    print('test input shape from origin data :', testInputs.shape, type(testInputs))

    #MaxTestRange = origin_data.shape[0]
    for i in range(INPUT_SIZE, MaxTestRange):
        X_test.append(testInputs[i-INPUT_SIZE:i, 0])
    X_test = np.array(X_test)
    #padding 7days zero data for test
    #paddingZeroLenght = origin_data.shape[0]-X_test.shape[0]- X_train.shape[0]- INPUT_SIZE + TestPredictDay
    #Zero = np.zeros([paddingZeroLenght,1, X_test.shape[1]], dtype = float)
    #print('Zero shape: ', Zero.shape )

    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    print('X test shape :', X_test.shape, type(X_test))


    #X_train + Xtest 
    #X_train_X_test = np.concatenate((X_train, X_test, Zero),axis=0)
    X_train_X_test = np.concatenate((X_train, X_test),axis=0)
    print('X_train + X_Test shape :', X_train_X_test.shape, type(X_train_X_test))
    #predict from after trained model

    hidden_state = None
    #load model
    model2 = torch.load(fileName)
    if torch.cuda.is_available() and EnableGPU:
        test_inputs = Variable(torch.from_numpy(X_train_X_test).float().cuda())
        print('test input shape befor test model :', test_inputs.shape, type(test_inputs))
        predicted_stock_price , b = model2(test_inputs, hidden_state)
        print('predict stock prince shape :', predicted_stock_price.shape, type(predicted_stock_price))
        predicted_stock_price = np.reshape(predicted_stock_price.cpu().detach().numpy(), (test_inputs.shape[0], 1))
    else:
        test_inputs = Variable(torch.from_numpy(X_train_X_test).float())
        print('test input shape befor test model :', test_inputs.shape, type(test_inputs))
        predicted_stock_price , b = model2(test_inputs, hidden_state)
        predicted_stock_price = np.reshape(predicted_stock_price.detach().numpy(), (test_inputs.shape[0], 1))
    
    return origin_data , predicted_stock_price

# For predict future n day 
def predictFuturePrice(trainScalar, origin_data, fileName):
    PrePredictOut =[]
    hidden_state = None
    for day in range (0, TestPredictDay + 1):
        print('Predict Day : ', day)
        if day != 0:
            previousInputs= origin_data[-INPUT_SIZE -1  :] # get last 61 days result 
            #previousInputs[-1] = PrePredictOut[-1]
            previousInputs = np.concatenate((previousInputs, PrePredictOut),axis=0) # add Predict n day out 
            #print('New Previous Input shape', previousInputs)
        else:
            previousInputs= origin_data[-INPUT_SIZE -1+day :] 

        previousInputs = previousInputs.reshape(-1, 1)
        previousInputs = trainScalar.transform(previousInputs)
        print('Previous input shape from origin data :', previousInputs.shape, type(previousInputs))
        TestData = []   
        if day != 0:
            for i in range(INPUT_SIZE, INPUT_SIZE+2):
                TestData.append(previousInputs[i-INPUT_SIZE+day: i +day, 0]) # shift day to 
        else:    
            for i in range(INPUT_SIZE, INPUT_SIZE+2):
                TestData.append(previousInputs[i-INPUT_SIZE:i, 0])

        TestData = np.array(TestData)
        print('TestData input shape :', TestData.shape, type(TestData))
        TestData = np.reshape(TestData, (TestData.shape[0], 1, TestData.shape[1]))
        print('TestData input after reshape :', TestData.shape, type(TestData))


        #load model
        model2 = torch.load(fileName)
        if torch.cuda.is_available() and EnableGPU:
            predict_inputs = Variable(torch.from_numpy(TestData).float().cuda())
            print('predict input shape befor test model :', predict_inputs.shape, type(predict_inputs))
            Npredicted_stock_price , hidden_state = model2(predict_inputs, hidden_state)
            print('predict stock Nprice shape :', Npredicted_stock_price.shape, type(Npredicted_stock_price))
            Npredicted_stock_price = np.reshape(Npredicted_stock_price.cpu().detach().numpy(), (predict_inputs.shape[0], 1))
            print('N predict stock price', Npredicted_stock_price)
            PredictOutputValue = trainScalar.inverse_transform(Npredicted_stock_price)
            #PrePredictOut.append(Npredicted_stock_price[-1:, 0])
            PrePredictOut.append(PredictOutputValue[-1:, 0])
            print('PrePredictOut :', PrePredictOut)
        else:
            predict_input = Variable(torch.from_numpy(TestData).float())
            print('test input shape befor test model :', predict_inputs.shape, type(predict_inputs))
            Npredicted_stock_price , hidden_state = model2(predict_inputs, hidden_state)
            Npredicted_stock_price = np.reshape(Npredicted_stock_price.detach().numpy(), (predict_input.shape[0], 1))
            PredictOutputValue = trainScalar.inverse_transform(Npredicted_stock_price)
            PrePredictOut.append(PredictOutputValue[-1:, 0])
            print('PrePredictOut :', PrePredictOut)


        print('Predict Output Value after Scalar: ', PredictOutputValue)
    return PrePredictOut



def runTrainPredict(StockDict, StockKey, StockColumn, fileName):
    print('Stock :', StockKey)
    resultEpoch = []
    resultLoss = []    
    X_train, y_train, train_data, test_data, TestStock, trainScalar, hidden_state =  TrainStockPrepare(StockDict,
                                                                                                       StockKey, 
                                                                                                       StockColumn) 
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

    if torch.cuda.is_available() and EnableGPU:
        model.cuda()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
     # loss function
    criterion = nn.MSELoss() 

    hidden_state = None
    StartTrainTime = datetime.datetime.now()
    LTSMStockTrain(X_train, y_train, hidden_state, model, criterion, optimiser,resultEpoch, resultLoss)

    #save model
    torch.save(model, fileName)
    StopTrainTime = datetime.datetime.now()- StartTrainTime

    StartTestTime = datetime.datetime.now()
    origin_data, predicted_stock_price = predictByTrainHistory(X_train, train_data, test_data, trainScalar, fileName)

    PrePredictOut = predictFuturePrice(trainScalar, origin_data, fileName)

    #invert scale to predict price
    predicted_stock_price = trainScalar.inverse_transform(predicted_stock_price)
    StopTestTime = datetime.datetime.now()- StartTestTime
    print('Predicted stock price shape:', predicted_stock_price.shape)

    real_stock_price_all = origin_data[INPUT_SIZE:]#np.concatenate((training_set[INPUT_SIZE:], real_stock_price))
    
    if torch.cuda.is_available() and EnableGPU:
        print('Stock :', StockKey)
        print("\n\r(GPU) Train Time : ", StopTrainTime.total_seconds(), "s")
        print("(GPU) Test Time :", StopTestTime.total_seconds() , "s")
        torch.cuda.empty_cache() 
    else:
        print('Stock :', StockKey)
        print("\n\r(CPU) Train Time : ", StopTrainTime.total_seconds(), "s")
        print("(CPU) Test Time :", StopTestTime.total_seconds() , "s")
    
    #plot Result
    print('Stock :', StockKey)
    plotLossResult(resultEpoch, resultLoss)
    plotPredictByHistory(real_stock_price_all, predicted_stock_price, TestStock)
    plotPredictFuturePrice(PrePredictOut)

    return PrePredictOut, resultEpoch, resultLoss
    #return origin_data, real_stock_price_all, predicted_stock_price, PrePredictOut, TestStock

#Main run for 8 stocks
PredictStock1Out, resultStock1Epoch, resultStock1Loss = runTrainPredict(RawStockList, RawStock1Key, AdjCloseIndex, 'trained1.pkl')
PredictStock2Out, resultStock2Epoch, resultStock2Loss = runTrainPredict(RawStockList, RawStock2Key, AdjCloseIndex, 'trained2.pkl')
PredictStock3Out, resultStock3Epoch, resultStock3Loss = runTrainPredict(RawStockList, RawStock3Key, AdjCloseIndex, 'trained3.pkl')
PredictStock4Out, resultStock4Epoch, resultStock4Loss = runTrainPredict(RawStockList, RawStock4Key, AdjCloseIndex, 'trained4.pkl')
PredictStock5Out, resultStock5Epoch, resultStock5Loss = runTrainPredict(RawStockList, RawStock5Key, AdjCloseIndex, 'trained5.pkl')
PredictStock6Out, resultStock6Epoch, resultStock6Loss = runTrainPredict(RawStockList, RawStock6Key, AdjCloseIndex, 'trained6.pkl')
PredictStock7Out, resultStock7Epoch, resultStock7Loss = runTrainPredict(RawStockList, RawStock7Key, AdjCloseIndex, 'trained7.pkl')
PredictStock8Out, resultStock8Epoch, resultStock8Loss = runTrainPredict(RawStockList, RawStock8Key, AdjCloseIndex, 'trained8.pkl')


#Plot Sumary all MSE 
plt.figure(figsize=(12,8))
plt.plot(resultStock1Epoch, resultStock1Loss, label = 'Stock1')
plt.plot(resultStock2Epoch, resultStock2Loss, label = 'Stock2')
plt.plot(resultStock3Epoch, resultStock3Loss, label = 'Stock3')
plt.plot(resultStock4Epoch, resultStock4Loss, label = 'Stock4')
plt.plot(resultStock5Epoch, resultStock5Loss, label = 'Stock5')
plt.plot(resultStock6Epoch, resultStock6Loss, label = 'Stock6')
plt.plot(resultStock7Epoch, resultStock7Loss, label = 'Stock7')
plt.plot(resultStock8Epoch, resultStock8Loss, label = 'Stock8')
plt.xlabel('Step (Training)')
plt.ylabel('Loss (%)')
plt.title('Loss Vs Number of Step')
plt.legend(loc='best')
plt.show()

#Plot Sumary all of PreictStock
plt.figure(figsize= (12,8))
plt.plot(PredictStock1Out[1:], label = 'Stock1')
plt.plot(PredictStock2Out[1:], label = 'Stock2')
plt.plot(PredictStock3Out[1:], label = 'Stock3')
plt.plot(PredictStock4Out[1:], label = 'Stock4')
plt.plot(PredictStock5Out[1:], label = 'Stock5')
plt.plot(PredictStock6Out[1:], label = 'Stock6')
plt.plot(PredictStock7Out[1:], label = 'Stock7')
plt.plot(PredictStock8Out[1:], label = 'Stock8')
#plt.xticks(range(0, TestStock.shape[0],20),TestStock.index[750+60::20], rotation=45)
plt.xlabel('Next Time (days)')
plt.ylabel('Predict Price')
#plt.yticks(np.arange(0, max(predictOut), 20)) 
plt.xticks(range(0, TestPredictDay),predictFutureDate ) 
plt.title('Predict N Future Days Price')
plt.legend(loc='best')
plt.show()







