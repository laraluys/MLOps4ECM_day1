import pandas as pd 
import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class Data_utils(object):
    def __init__(self):
        """Initialize data_utils class given the arguments from inference_main.py parse_args() function
        """

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.cur_data = None
        
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        self.cur_dataloader = None

    def split_data(self):
        data = pd.read_csv("dataset/water_potability.csv", index_col=None)
        # Split data depending on hardness in reference and current data
        data_reference = data.loc[(data['Hardness'] <= 186) ]
        data_current = data.loc[(data['Hardness'] > 186)]
        data_reference.to_csv('dataset/dataset_reference_Evidently.csv', sep=',', encoding='utf-8')
        data_current.to_csv('dataset/dataset_current_Evidently.csv', sep=',', encoding='utf-8')
        return data_reference


    def data_cleaning(self, dataset):
        # replace NaN data with mean of the column
        mean = dataset.mean()
        dataset.fillna(mean,inplace=True)

        # use min-max normalization
        for column in dataset.columns: 
            dataset[column] = (dataset[column] - dataset[column].min()) / (dataset[column].max() - dataset[column].min())
        
        return dataset


    def pandas_to_tensor(self, dataset):
        df_values = dataset.drop(columns=['Potability'])
        df_labels = dataset['Potability']

        # not_potable = []
        # for row in df_labels:
        #     not_potable.append(1-row)

        tensor_values = torch.tensor(df_values.to_numpy(), dtype=torch.float32) 
        tensor_labels = torch.tensor(df_labels.to_numpy(), dtype=torch.float32)   
        # tensor_labels_1 = torch.tensor(df_labels.to_numpy(), dtype=torch.float32).reshape(-1,1)
        # tensor_labels_2 = torch.tensor(not_potable, dtype=torch.float32).reshape(-1,1)

        # tensor_labels = torch.cat((tensor_labels_1,tensor_labels_2), -1)

        return tensor_values, tensor_labels

    def create_dataloader(self, data, labels, batch_size):
        dataset = TensorDataset(data, labels)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return data_loader

    def load_train_data(self, batch_size):
        existing_data = os.listdir("dataset/")
        if "dataset_reference.csv" in existing_data:
            data_reference = pd.read_csv("dataset/dataset_reference_Evidently.csv", index_col=None)
        else:
            data_reference = self.split_data()
        print(len(data_reference))
        # replace NaN data with mean of the column
        data_reference = self.data_cleaning(data_reference)    
        train_values, train_labels = self.pandas_to_tensor(data_reference)

        self.train_dataloader = self.create_dataloader(train_values, train_labels, batch_size)
        self.train_data = [train_values, train_labels]

    def load_test_data(self, batch_size):
    
        data_current = pd.read_csv("dataset/dataset_current_Evidently.csv", index_col=None)
        data_current = self.data_cleaning(data_current)
        
        cur_values, cur_labels = self.pandas_to_tensor(data_current)
        self.cur_dataloader = self.create_dataloader(cur_values, cur_labels, batch_size)

        self.cur_data = [cur_values, cur_labels]
