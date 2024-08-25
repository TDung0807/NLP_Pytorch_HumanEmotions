import pandas as pd
import torch
import torch.nn as nn
import re
class DataProcessing:
    def __init__(self, data):
        self.data = data
    def tokenizedWord(self,row):
        self.data["tokenized"] = self.data[row].apply(lambda x: re.findall(r'\b\w+\b',x.lower()))
        self.stopWord = pd.read_csv('stopword.csv')
        arr = self.stopWord.transpose().values
        self.data["tokenized"] = self.data['tokenized'].apply(lambda x: self.remove_stopWord(x,arr))
        print(self.data.head())
        
    def remove_stopWord(self,dataRow,stopWords):
        for i in dataRow:
            if i in stopWords:
                dataRow.remove(i)
        return dataRow
    
# Example usage
df = pd.read_csv("./data/train.csv")
dp = DataProcessing(df)
dp.tokenizedWord('message')