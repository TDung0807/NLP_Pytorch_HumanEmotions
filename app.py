import pandas as pd
import torch
import torch.nn as nn
df_train= pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")
df_val = pd.read_csv("./data/val.csv")


print(df_train.head())