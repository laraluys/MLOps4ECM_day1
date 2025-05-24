import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def split_in_sequences(data, seq_len):
    dataframe_dict = {}
    total_len = len(data)
    rest = total_len % seq_len
    data.drop(data.tail(rest).index,inplace=True)
    for name, values in data.items():
        dataframe_dict[name] = data[name].to_numpy().reshape(-1, seq_len).tolist()
    dataframe_split = pd.DataFrame(dataframe_dict)
    return dataframe_split

def create_dataloader(dataframe_train, batch_size):
    pressure = torch.tensor(dataframe_train['value'].to_list(), dtype=torch.float32)
    dataset = TensorDataset(pressure)
    data_loader = DataLoader(dataset, batch_size, shuffle=False)
    return data_loader

def split_train_test_dataset(dataset):
    train_dataset, rest_dataset = train_test_split(dataset, shuffle=False, test_size=0.2)
    val_dataset, test_dataset = train_test_split(rest_dataset, shuffle=False, test_size=0.5)
    return train_dataset, val_dataset, test_dataset

