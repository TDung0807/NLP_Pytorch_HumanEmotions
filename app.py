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


X = df_train['frequency'].tolist()
y = df_train['emotion'].tolist()  
unique_labels = list(set(y))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

df_train['label'] = df_train['emotion'].map(label_to_index)


