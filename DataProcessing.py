import pandas as pd
import re
from sklearn.feature_extraction.text import HashingVectorizer

class DataProcessing:
    def __init__(self, data):
        self.data = data
        
    def tokenizedWord(self,row):
        self.data["tokenized"] = self.data[row].apply(lambda x: re.findall(r'\b\w+\b',x.lower()))
        self.stopWord = pd.read_csv('stopword.csv')
        arr = self.stopWord.transpose().values
        self.data["tokenized"] = self.data['tokenized'].apply(lambda x: self.remove_stopWord(x,arr))
        pass
    
    def wordFrequency(self):
        vectorizer = HashingVectorizer(n_features=6)
        self.data["text"] = self.data["tokenized"].apply(lambda x: ' '.join(x))
        vector = vectorizer.transform(self.data['text'])
        self.data["frequency"] = list(vector.toarray())
        print(self.data)
        pass
    
    def remove_stopWord(self, dataRow, stopWords):
        return [word for word in dataRow if word not in stopWords]
    
    def main(self,row):
        self.tokenizedWord(row)
        self.wordFrequency()
        return self.data