import os
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from torch import FloatTensor, tensor
from torch.cuda import FloatTensor as GPUFloatTensor
from tqdm import tqdm
from torch.utils.data import Dataset
from ..NN.Helpers import DEVICE

# Folder name will correspond to index of sample
class SkinDataset(Dataset):
    def __init__(
        self,
        scaler,
        signal_folder="D:/SamplingResults2",
        sample_file="../Data/newSamples.pkl",
        runs=range(65535),
        steps=128
        ):
        # Load both disp1 and disp2 from each folder
        # Folders ordered according to index of sample
        self.input = []
        self.output = []
        
        with open(f"{sample_file}", "rb") as f:
             samples = pickle.load(f)
        
        for run in tqdm(runs):
            inp = []
            fail = False
            
            files = os.listdir(f"{signal_folder}/{run}/")
            
            if files != ['Disp1.csv', 'Disp2.csv']:
                continue
            
            for file in files:
                a = pd.read_csv(f"{signal_folder}/{run}/{file}")
                a.rename(columns = {'0':'x', '0.1': 'y'}, inplace = True)
                
                # Skip if unconverged
                if a['x'].max() != 7.0:
                    fail = True
                    break

                # Interpolate curve for consistent x values
                xNew = np.linspace(0, 7, num=steps, endpoint=False)
                interped = interp1d(a['x'], a['y'], kind='cubic', fill_value="extrapolate")(xNew)        
                
                inp.append(interped.astype("float32"))
            
            if not fail:
                if len(inp) != 2:
                    raise Exception("sdf")

                self.input.append(inp)
                self.output.append(samples[int(run)])
        
        # Normalise output variables
        self.output = scaler.fit_transform(self.output)
        
        self.output = tensor(self.output).type(
            FloatTensor if DEVICE == 'cpu' else GPUFloatTensor
        )
        self.input = tensor(np.array(self.input)).type(
            FloatTensor if DEVICE == 'cpu' else GPUFloatTensor
        )
        
    def __len__(self):
        return len(self.output)
    
    def __getitem__(self, idx):
        sample = {"input": self.input[idx], "output": self.output[idx]}
        return sample

