import pickle
import TweetTrainProgram
import StockPredictionProgram

tweetTrainObtainedResult = []
lstmTrainObtainedResult = []
combinedStockPredictionResult = []
combinedWeightTweet = 0.8
combinedWeightLSTM = 0.2

def saveResultToPKLFile():
    fw = open('19035814g.pkl','wb')
    pickle.dump(combinedStockPredictionResult, fw)
    fw.close()
    fr = open('19035814g.pkl','rb')
    loadFileData = pickle.load(fr)
    fr.close()
    print('loadFileData:',loadFileData)

def main():    
    global tweetTrainObtainedResult
    global lstmTrainObtainedResult
    global combinedStockPredictionResult
    tweetTrainObtainedResult = TweetTrainProgram.returnTweetTrainProgramStockPriceResult()
    lstmTrainObtainedResult = StockPredictionProgram.returnLSTMTrainProgramStockPriceResult()
    for i in range(len(lstmTrainObtainedResult)):
        combinedStockPredictionResult.append(tweetTrainObtainedResult[i]*combinedWeightTweet+lstmTrainObtainedResult[i]*combinedWeightLSTM)
    print("combinedStockPredictionResult:",combinedStockPredictionResult)
    saveResultToPKLFile()
    
if __name__== "__main__":
  main()