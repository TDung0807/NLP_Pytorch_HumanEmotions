# main.py
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from DataProcessing import DataProcessing  # Import the DataProcessing class

# Load your data
df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")
df_val = pd.read_csv("./data/val.csv")

process = DataProcessing(df_train)
df_train = process.main('message')

print(df_train.head())

X = df_train['frequency'].tolist()
y = df_train['emotion'].tolist()  

