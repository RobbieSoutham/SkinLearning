import os
import pickle
from ..Dataset.SkinDataset import SkinDataset
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from .Misc import running_in_notebook

"""
    Creates the data set from filtered samples
    
    Returns
        dataset (SkinDataset)
            The generated dataset.
        scaler (MinMaxScaler)
            The scaler used for normalising outputs.
"""
def get_dataset(**kwargs):
    # Ensure path is correct
    if running_in_notebook():
        filtered_file = "../Data/filtered.pkl"
        kwargs['sample_file'] = "../Data/newSamples.pkl"
        kwargs['signal_folder'] = "D:/SamplingResults2/"
    else:
        filtered_file = 'Data/filtered.pkl'

    # Get filtered dataset indeces
    if not 'runs' in kwargs.keys():
        with open(filtered_file, 'rb') as f:
            runs = pickle.load(f)

        kwargs['runs'] = runs

    scaler = MinMaxScaler()
    dataset = SkinDataset(scaler=scaler, **kwargs)

    return dataset, scaler

"""
    Creates a train/test split from the given data

    Parameters:
        dataset (Dataset):
            The dataset to split.
        p1 (float):
            Fraction of the dataset used for testing.
        batch_size:
            The batch size of the data loaders.
    
    Returns
        train (DataLoader):
            The training set.
        test (DataLoader):
            The validation set.
"""
def get_split(dataset, p1=0.8, batch_size=32):
    train_n = int(p1 * len(dataset))
    test_n = len(dataset) - train_n
    train_set, test_set = random_split(dataset, [train_n, test_n])

    return DataLoader(train_set, batch_size=batch_size, shuffle=True), \
        DataLoader(test_set, batch_size=batch_size, shuffle=True)