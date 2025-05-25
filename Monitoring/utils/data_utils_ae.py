import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def split_in_sequences(data, seq_len):
    """Split a full dataframe into sections of length seq_len.
      Instead of one value per row you will have seq_len values per row which is a sample of your dataset. 
      These are going to be the input of your machine learning model. """
    dataframe_dict = {}
    total_len = len(data)
    rest = total_len % seq_len
    data.drop(data.tail(rest).index,inplace=True)
    for name, values in data.items():
        dataframe_dict[name] = data[name].to_numpy().reshape(-1, seq_len).tolist()
    dataframe_split = pd.DataFrame(dataframe_dict)
    return dataframe_split

def create_dataloader(dataframe_train, batch_size):
    """ The dataloader will make your dataframe iterable for training. 
    The amount of input samples given at a time will be the batch_size.
    """
    pressure = torch.tensor(dataframe_train['value'].to_list(), dtype=torch.float32)
    dataset = TensorDataset(pressure)
    data_loader = DataLoader(dataset, batch_size, shuffle=False)
    return data_loader

def split_train_test_dataset(dataset):
    "Split your dataframe into a training, validation and test dataframe"
    train_dataset, rest_dataset = train_test_split(dataset, shuffle=False, test_size=0.2)
    val_dataset, test_dataset = train_test_split(rest_dataset, shuffle=False, test_size=0.5)
    return train_dataset, val_dataset, test_dataset

