import pandas as pd
import torch
import torch.nn as nn
import re
class DataProcessing:
    def __init__(self, data):
        self.data = data
    def tokenizedWord(self,row):
        self.data["tokenized"] = self.data[row].apply(lambda x: re.findall(r'\b\w+\b',x.lower()))
        print(self.data.head())

# Example usage
df = pd.read_csv("./data/train.csv")
dp = DataProcessing(df)
dp.tokenizedWord('message')