from collections import Counter
import os
import json
import math
import torch
import pickle
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

tweetTrainDataFrame = pd.DataFrame(columns = ['Date', 'Dataline', 'Value'])
tweetTestDataFrame = pd.DataFrame(columns = ['Date', 'Dataline'])
datalinesTrain = []
datalinesTest = []
vocabularyListTrain = []
vocabularyListTest = []
bowListTrain = []
bowListTest = []
vocabularyTopsWords = []
changePredictionResultList = []
adjustPriceList = [] 
stockPricePredictionResultList = []
resultEpochTweet = []
resultLossTweet = []  

# RNN hyperparameters 1
input_size_tweet=1000
output_size_tweet=1
hidden_dim=100
batch_size = 1
n_layers_tweet=5
lr_tweet=0.004
n_steps=60
print_every=10
NoOfPredictDay = 7

# Declare Train stock  path 
PreTrainStock1 = 'stock_dataset_v3/preprocess_price_train/1_p_price_train.txt'
PreTrainStock2 = 'stock_dataset_v3/preprocess_price_train/2_p_price_train.txt'
PreTrainStock3 = 'stock_dataset_v3/preprocess_price_train/3_p_price_train.txt'
PreTrainStock4 = 'stock_dataset_v3/preprocess_price_train/4_p_price_train.txt'
PreTrainStock5 = 'stock_dataset_v3/preprocess_price_train/5_p_price_train.txt'
PreTrainStock6 = 'stock_dataset_v3/preprocess_price_train/6_p_price_train.txt'
PreTrainStock7 = 'stock_dataset_v3/preprocess_price_train/7_p_price_train.txt'
PreTrainStock8 = 'stock_dataset_v3/preprocess_price_train/8_p_price_train.txt'
preTrainStock1 = pd.read_csv(PreTrainStock1, header=None, delimiter='\t',names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],  index_col="Date") 
preTrainStock2 = pd.read_csv(PreTrainStock2, header=None, delimiter='\t',names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],  index_col="Date") 
preTrainStock3 = pd.read_csv(PreTrainStock3, header=None, delimiter='\t',names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],  index_col="Date") 
preTrainStock4 = pd.read_csv(PreTrainStock4, header=None, delimiter='\t',names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],  index_col="Date")
preTrainStock5 = pd.read_csv(PreTrainStock5, header=None, delimiter='\t',names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],  index_col="Date") 
preTrainStock6 = pd.read_csv(PreTrainStock6, header=None, delimiter='\t',names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],  index_col="Date") 
preTrainStock7 = pd.read_csv(PreTrainStock7, header=None, delimiter='\t',names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],  index_col="Date") 
preTrainStock8 = pd.read_csv(PreTrainStock8, header=None, delimiter='\t',names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"], index_col="Date") 
tweetTrainFile1 = 'stock_dataset_v3/tweet_train/1_tweet_train/'
tweetTrainFile2 = 'stock_dataset_v3/tweet_train/2_tweet_train/'
tweetTrainFile3 = 'stock_dataset_v3/tweet_train/3_tweet_train/'
tweetTrainFile4 = 'stock_dataset_v3/tweet_train/4_tweet_train/'
tweetTrainFile5 = 'stock_dataset_v3/tweet_train/5_tweet_train/'
tweetTrainFile6 = 'stock_dataset_v3/tweet_train/6_tweet_train/'
tweetTrainFile7 = 'stock_dataset_v3/tweet_train/7_tweet_train/'
tweetTrainFile8 = 'stock_dataset_v3/tweet_train/8_tweet_train/'
tweetTestFile1 = 'stock_dataset_v3/tweet_test/1/'
tweetTestFile2 = 'stock_dataset_v3/tweet_test/2/'
tweetTestFile3 = 'stock_dataset_v3/tweet_test/3/'
tweetTestFile4 = 'stock_dataset_v3/tweet_test/4/'
tweetTestFile5 = 'stock_dataset_v3/tweet_test/5/'
tweetTestFile6 = 'stock_dataset_v3/tweet_test/6/'
tweetTestFile7 = 'stock_dataset_v3/tweet_test/7/'
tweetTestFile8 = 'stock_dataset_v3/tweet_test/8/'
RawStock1 = 'stock_dataset_v3/raw_price_train/1_r_price_train.csv'
RawStock2 = 'stock_dataset_v3/raw_price_train/2_r_price_train.csv'
RawStock3 = 'stock_dataset_v3/raw_price_train/3_r_price_train.csv'
RawStock4 = 'stock_dataset_v3/raw_price_train/4_r_price_train.csv'
RawStock5 = 'stock_dataset_v3/raw_price_train/5_r_price_train.csv'
RawStock6 = 'stock_dataset_v3/raw_price_train/6_r_price_train.csv'
RawStock7 = 'stock_dataset_v3/raw_price_train/7_r_price_train.csv'
RawStock8 = 'stock_dataset_v3/raw_price_train/8_r_price_train.csv'
rawStock1 = pd.read_csv(RawStock1, index_col="Date", parse_dates=True) 
rawStock2 = pd.read_csv(RawStock2, index_col="Date", parse_dates=True) 
rawStock3 = pd.read_csv(RawStock3, index_col="Date", parse_dates=True) 
rawStock4 = pd.read_csv(RawStock4, index_col="Date", parse_dates=True) 
rawStock5 = pd.read_csv(RawStock5, index_col="Date", parse_dates=True) 
rawStock6 = pd.read_csv(RawStock6, index_col="Date", parse_dates=True) 
rawStock7 = pd.read_csv(RawStock7, index_col="Date", parse_dates=True) 
rawStock8 = pd.read_csv(RawStock8, index_col="Date", parse_dates=True) 

predictFutureDate = []
predictFutureDate.append('2015-12-21')
predictFutureDate.append('2015-12-22')
predictFutureDate.append('2015-12-23')
predictFutureDate.append('2015-12-24')
predictFutureDate.append('2015-12-28')
predictFutureDate.append('2015-12-29')
predictFutureDate.append('2015-12-30')

def unique_list(inList1):   
    list_set = set(inList1) 
    unique_list = (list(list_set))
    return unique_list

def readTrainFiles(filePathTrain):
    folderPathTrain = filePathTrain
    filenamesTrain = os.listdir(folderPathTrain)
    for filenameTrain in filenamesTrain:
        with open(folderPathTrain+filenameTrain,'r') as fileInputTrain:
            for lineTrain in fileInputTrain:
                datalinesTrain.append([json.loads(lineTrain),datetime.datetime.strptime(json.loads(lineTrain)['created_at'], '%a %b %d %H:%M:%S %z %Y').strftime("%Y-%m-%d")])

def readTestFiles(filePathTest):
    folderPathTest = filePathTest
    filenamesTest = os.listdir(folderPathTest)
    for filenameTest in filenamesTest:
        with open(folderPathTest+filenameTest,'r') as fileInputTest:
            for lineTest in fileInputTest:
                datalinesTest.append([json.loads(lineTest),datetime.datetime.strptime(json.loads(lineTest)['created_at'], '%a %b %d %H:%M:%S %z %Y').strftime("%Y-%m-%d")])


def buildTrainDataSetFrame(preTrainStockNo):
    previous_document_date_time = ''
    tempWordList = []
    for jsonline in datalinesTrain:
        document_date_time = datetime.datetime.strptime(jsonline[0]['created_at'], '%a %b %d %H:%M:%S %z %Y').strftime("%Y-%m-%d")
        if document_date_time in preTrainStockNo.index:
            if previous_document_date_time == '':
                previous_document_date_time = document_date_time
            if previous_document_date_time != document_date_time:
                tempWordList = unique_list(tempWordList)
                tweetTrainDataFrame.loc[len(tweetTrainDataFrame)] = [previous_document_date_time, tempWordList, preTrainStockNo.loc[previous_document_date_time,"Adj Close"]]
                previous_document_date_time = document_date_time
                tempWordList = []
    
            for word in jsonline[0]['text']:
                vocabularyListTrain.append(word)
                tempWordList.append(word)
    #last loop write       
    if document_date_time in preTrainStockNo.index:    
        tempWordList = unique_list(tempWordList)
        tweetTrainDataFrame.loc[len(tweetTrainDataFrame)] = [previous_document_date_time, tempWordList, preTrainStockNo.loc[previous_document_date_time,"Adj Close"]]     

def buildTestDataSetFrame():
    previous_document_date_time = ''
    tempWordList = []
    for jsonline in datalinesTest:
        document_date_time = datetime.datetime.strptime(jsonline[0]['created_at'], '%a %b %d %H:%M:%S %z %Y').strftime("%Y-%m-%d")
        if document_date_time in predictFutureDate:
            if previous_document_date_time == '':
                previous_document_date_time = document_date_time
            if previous_document_date_time != document_date_time:
                tempWordList = unique_list(tempWordList)
                tweetTestDataFrame.loc[len(tweetTestDataFrame)] = [previous_document_date_time, tempWordList]
                previous_document_date_time = document_date_time
                tempWordList = []
    
            for word in jsonline[0]['text']:
                vocabularyListTest.append(word)
                tempWordList.append(word)
    #last loop write            
    tempWordList = unique_list(tempWordList)
    tweetTestDataFrame.loc[len(tweetTestDataFrame)] = [previous_document_date_time, tempWordList]  

def buildBagOfWordsTrain():
    #input_size_tweet set 1000 top words
    global vocabularyTopsWords
    vocabularyTopsWords = []
    vocabularyTops = Counter(vocabularyListTrain).most_common(input_size_tweet)
    vocabularyTopsWords = [item[0] for item in vocabularyTops]
    for index, row in tweetTrainDataFrame.iterrows():
        bowVec = []
        for token in vocabularyTopsWords:
            if token in tweetTrainDataFrame.iloc[index,1]:
                bowVec.append(1)
            else:
                bowVec.append(0)
        bowListTrain.append(bowVec)
        
def buildBagOfWordsTest():
    #use vocabularyTopsWords from train
    for index, row in tweetTestDataFrame.iterrows():
        bowVec = []
        for token in vocabularyTopsWords:
            if token in tweetTestDataFrame.iloc[index,1]:
                bowVec.append(1)
            else:
                bowVec.append(0)
        bowListTest.append(bowVec)

def dataExtractionForTrainStockNo(trainFileNoInput,preTrainStockNoInput):
    global datalinesTrain
    global vocabularyListTrain
    global bowListTrain
    global tweetTrainDataFrame
    #Init train dataset
    datalinesTrain = []
    vocabularyListTrain = []
    bowListTrain = []
    tweetTrainDataFrame = pd.DataFrame(columns = ['Date', 'Dataline', 'Value'])
    #Train data extraction process
    readTrainFiles(trainFileNoInput)
    datalinesTrain = sorted(datalinesTrain,key=lambda x:x[1])
    buildTrainDataSetFrame(preTrainStockNoInput)
    buildBagOfWordsTrain();
    
def dataExtractionForTestStockNo(testFileNoInput):
    global datalinesTest
    global vocabularyListTest
    global bowListTest
    global tweetTestDataFrame
    #Init test dataset
    datalinesTest = []
    vocabularyListTest = []
    bowListTest = []
    tweetTestDataFrame = pd.DataFrame(columns = ['Date', 'Dataline'])
    #Test data extraction process
    readTestFiles(testFileNoInput)
    datalinesTest = sorted(datalinesTest,key=lambda x:x[1])
    buildTestDataSetFrame()
    buildBagOfWordsTest();

##############################################################################

class RNN(nn.Module):
    def __init__(self, input_size_tweet, output_size_tweet, hidden_dim, n_layers_tweet):
        super(RNN, self).__init__()
        
        self.hidden_dim=hidden_dim
        #self.lstm = nn.LSTM(input_size_tweet, hidden_dim, n_layers_tweet, batch_first=True)
        self.embedding = nn.Embedding(input_size_tweet, hidden_dim)
        self.rnn = nn.RNN(input_size_tweet, hidden_dim, n_layers_tweet, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size_tweet)

    def forward(self, x, hidden):
        r_out, hidden = self.rnn(x, hidden)
        #r_out, hidden = self.lstm(x, hidden)
        #hidden_size = hidden[-1].size(-1)  # for LSTM
        #r_out = r_out.view(-1, hidden_size) # for LSTM
        output = self.fc(r_out)
        return output, hidden

def train(rnn, n_steps, print_every, inputsTrainInput, labelsTrainInput):    
    # MSE loss and Adam optimizer with a learning rate
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr_tweet)
    hidden = None
    loss_val = []
    for batch_i, step in enumerate(range(n_steps)):
        predictionTrain, hidden = rnn(inputsTrainInput, hidden)
        #hidden = hidden.data
        #hidden = hidden # for LSTM
        loss = criterion(predictionTrain, labelsTrainInput)
        loss_val.append(loss.item())

        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # for LSTM
        #loss.backward()
        optimizer.step()

        #if batch_i%print_every == 0 or batch_i == n_steps-1:
            #print("Epoch: {0}/{1}".format(batch_i+1, n_steps)+'  Loss:', loss.item())
            #print(predictionTrain.data[0])
        resultEpochTweet.append(batch_i)
        resultLossTweet.append(loss.item())
            
        #if batch_i == n_steps-1:
            #print('At train - n_steps-1 ')
            #predictionOutput = predictionTrain.tolist()
            #print('predictionOutput ',predictionOutput)  
    return rnn

def runRNNModel():
    inputXTrain = bowListTrain
    inputYTrain = tweetTrainDataFrame.iloc[:,2]
    inputsTrain = Variable(torch.Tensor(inputXTrain).float())
    inputsTrain = np.reshape(inputsTrain, (inputsTrain.shape[0], 1, inputsTrain.shape[1]))
    labelsTrain = Variable(torch.LongTensor(inputYTrain).float())
    
    inputXTest = bowListTest
    inputsTest = Variable(torch.Tensor(inputXTest).float())
    inputsTest = np.reshape(inputsTest, (inputsTest.shape[0], 1, inputsTest.shape[1]))

    rnn = RNN(input_size_tweet, output_size_tweet, hidden_dim, n_layers_tweet)
    print(rnn)
    trained_rnn_model = train(rnn, n_steps, print_every, inputsTrain, labelsTrain)
    
    hidden = None 
    predictOnTestResult = trained_rnn_model(inputsTest, hidden)
    
    for i in range(7):
        if(i<len(predictOnTestResult[0])):
            changePredictionResultList.append(predictOnTestResult[0][i][0][0].item())
        else:
            changePredictionResultList.append(0)
    
def plotTrainModelLossGraph():
    plt.figure(figsize=(12,8))
    plt.plot(resultEpochTweet, resultLossTweet)
    plt.xlabel('Step (Training)')
    plt.ylabel('Loss (%)')
    plt.title('Loss Vs Number of Step')
    plt.show()
    
def printChangeResultList():
    print("")
    print("Predict 7 days change result:")
    print(changePredictionResultList)
    print("")

def runMLForStocks(tweetTrainFileNo,preTrainStockNo,tweetTestFileNo,stockNo):
    global vocabularyTopsWords
    vocabularyTopsWords = []
    #Run Train and Test data extraction step
    dataExtractionForTrainStockNo(tweetTrainFileNo,preTrainStockNo)
    dataExtractionForTestStockNo(tweetTestFileNo)
    print("")
    print("Tweet Model Training for Stock",stockNo,":")
    #Run RNN Model step
    runRNNModel()
    
#########################################################################

def readAdjustPriceList():
    global adjustPriceList
    adjustPriceList = []  
    lastAvailableDate = '2015-12-18'
    adjCloseLabel = 'Adj Close'
    adjustPriceList.append(rawStock1.loc[lastAvailableDate,adjCloseLabel])
    adjustPriceList.append(rawStock2.loc[lastAvailableDate,adjCloseLabel][0])
    adjustPriceList.append(rawStock3.loc[lastAvailableDate,adjCloseLabel])
    adjustPriceList.append(rawStock4.loc[lastAvailableDate,adjCloseLabel])
    adjustPriceList.append(rawStock5.loc[lastAvailableDate,adjCloseLabel])
    adjustPriceList.append(rawStock6.loc[lastAvailableDate,adjCloseLabel][0])
    adjustPriceList.append(rawStock7.loc[lastAvailableDate,adjCloseLabel][0])
    adjustPriceList.append(rawStock8.loc[lastAvailableDate,adjCloseLabel])
    
def convertToStockPricePredictResultList():
    for i in range(NoOfPredictDay*8):
        if (i % NoOfPredictDay == 0) :
            stockPricePredictionResultList.append(adjustPriceList[math.floor(i/NoOfPredictDay)])
        else:
            if(changePredictionResultList[i]!=0):
                stockPricePredictionResultList.append(((changePredictionResultList[i]-changePredictionResultList[i-1])*100)+stockPricePredictionResultList[i-1])       
            else:
                stockPricePredictionResultList.append(adjustPriceList[math.floor(i/NoOfPredictDay)])
    
def printPredictValueResultList():
    print("")
    print("Tweet predict 7 days stock price :")
    print(stockPricePredictionResultList)
    print("")

def runTweetTrainProgram():
    #Main run for 8 stocks
    runMLForStocks(tweetTrainFile1,preTrainStock1,tweetTestFile1,1)
    runMLForStocks(tweetTrainFile2,preTrainStock2,tweetTestFile2,2)
    runMLForStocks(tweetTrainFile3,preTrainStock3,tweetTestFile3,3)
    runMLForStocks(tweetTrainFile4,preTrainStock4,tweetTestFile4,4)
    runMLForStocks(tweetTrainFile5,preTrainStock5,tweetTestFile5,5)
    runMLForStocks(tweetTrainFile6,preTrainStock6,tweetTestFile6,6)
    runMLForStocks(tweetTrainFile7,preTrainStock7,tweetTestFile7,7)
    runMLForStocks(tweetTrainFile8,preTrainStock8,tweetTestFile8,8)
    #plotTrainModelLossGraph()
    #printChangeResultList()   
    readAdjustPriceList()
    convertToStockPricePredictResultList()
    printPredictValueResultList()

#Run Main Tweet Train Program
def returnTweetTrainProgramStockPriceResult():
    runTweetTrainProgram()
    return stockPricePredictionResultList


