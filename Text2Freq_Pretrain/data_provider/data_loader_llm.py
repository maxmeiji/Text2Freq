import json
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, root, file, type):
        file = os.path.join(root, file)
        if type == "train":
            file_name = file + '_train_llm.json'
        elif type == "val":
            file_name = file + '_val_llm.json'
        elif type == "test":
            file_name = file + '_test_llm.json'
        else:
            file_name = file
        with open(file_name, 'r') as file:
            self.data = json.load(file)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_entry = self.data[idx]
        annotations = data_entry["series_description"]

        inputs = annotations
        # instance normalization
        series_str = data_entry["time_series"]
        series = list(map(float, series_str.split(',')))
        mean = np.mean(series)
        std = np.std(series)
        if std > 0:  # Avoid division by zero
            series = (series - mean) / std
        
        """
            print(f"Type of inputs: {type(inputs)}")
            print(f"Type of series: {type(series)}")
            Type of inputs: <class 'transformers.tokenization_utils_base.BatchEncoding'>
            Type of series: <class 'list'>
        """
        series_tensor = torch.tensor(series, dtype=torch.float32)
        return inputs, series_tensor

def create_data_loader(root, file, type, batch_size=32, shuffle=True):
    dataset = TimeSeriesDataset(root, file, type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

"""
if __name__ == "__main__":
    train_loader=create_data_loader('./processed_data/Pretraining', 'Weekly_half', "test", 1, shuffle=False)
    i = 1
    for input, series in train_loader:
        
        if(len(series[0])!=48):
            print(i, len(series[0]))
        i = i +1
"""
