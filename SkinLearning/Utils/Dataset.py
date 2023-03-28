import os
import pickle

import numpy as np
from SkinLearning.Dataset.SkinDataset import SkinDataset
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler

"""
    Creates the data set from filtered samples
    Returns the dataset and the scaler
"""
def getDataset(**kwargs):
    # Get filtered data
    if not 'runs' in kwargs.keys():
        with open("../Data/filtered.pkl", "rb") as f:
            runs = pickle.load(f)

        kwargs['runs'] = runs

    scaler = MinMaxScaler()
    dataset = SkinDataset(scaler=scaler, **kwargs)

    return dataset, scaler

"""
    Creates a train/test split from the given data
    Returns train and test data loaders
"""
def getSplit(dataset, p1=0.8, batch_size=32):
    train_n = int(p1 * len(dataset))
    test_n = len(dataset) - train_n
    train_set, test_set = random_split(dataset, [train_n, test_n])

    return DataLoader(train_set, batch_size=batch_size, shuffle=True), \
        DataLoader(test_set, batch_size=batch_size, shuffle=True)