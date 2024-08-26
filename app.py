# main.py
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from DataProcessing import DataProcessing  # Import the DataProcessing class
from torch.utils.data import DataLoader, TensorDataset
from ModelNN import ModelNN 

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


y = df_train['label'].tolist()

X_tensors = torch.tensor(X)
y_tensors = torch.tensor(y)

dataset = TensorDataset(X_tensors, y_tensors)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define model parameters
input_size = len(X_tensors[0])  # Number of features
hidden_size = 64  # Size of the hidden layer
output_size = len(set(y))  # Number of classes

model = ModelNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)