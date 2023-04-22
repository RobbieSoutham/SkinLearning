import os
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from torch import FloatTensor, tensor
from torch.cuda import FloatTensor as GPUFloatTensor
from tqdm import tqdm
from torch.utils.data import Dataset
import pywt
from scipy.stats import skew, kurtosis
from scipy.special import entr
from ..NN.Helpers import DEVICE

# Folder name will correspond to index of sample
class SkinDataset(Dataset):
    def __init__(
        self,
        scaler,
        signal_folder='D:/SamplingResults2',
        sample_file='Data/newSamples.pkl',
        runs=range(65535),
        steps=128,
        extraction_args=None
        ):
        # Load both disp1 and disp2 from each folder
        # Folders ordered according to index of sample
        self.input = []
        self.output = []

        with open(f'{sample_file}', 'rb') as f:
             samples = pickle.load(f)
        
        for run in tqdm(runs):
            inp = []
            fail = False
            
            files = os.listdir(f'{signal_folder}/{run}/')
            
            if files != ['Disp1.csv', 'Disp2.csv']:
                continue
            
            for file in files:
                a = pd.read_csv(f'{signal_folder}/{run}/{file}')
                a.rename(columns = {'0':'x', '0.1': 'y'}, inplace = True)

                # Interpolate curve for consistent x values
                xNew = np.linspace(0, 7, num=steps, endpoint=False)
                interped = interp1d(a['x'], a['y'], kind='cubic', fill_value='extrapolate')(xNew)        
                
                inp.append(interped.astype('float32'))
            

                self.input.append(inp)
                self.output.append(samples[int(run)])
        
        # Normalise output variables
        self.output = scaler.fit_transform(self.output)
        
        # Perform WPD if given
        if extraction_args:
            extracted_features = []
            for signals in self.input:
                extraction_args['signals'] = signals
                extracted_features.append(waveletExtraction(**extraction_args))
            self.input = extracted_features
            del extracted_features

        self.output = tensor(self.output).type(
            FloatTensor if DEVICE == 'cpu' else GPUFloatTensor
        )
        self.input = tensor(np.array(self.input)).type(
            FloatTensor if DEVICE == 'cpu' else GPUFloatTensor
        )
        
    def __len__(self):
        return len(self.output)
    
    def __getitem__(self, idx):
        sample = {'input': self.input[idx], 'output': self.output[idx]}
        return sample

def waveletExtraction(
    signals,
    method,
    combined=False,
    wavelet='db4',
    level=6,
    combine_method='concatenate',
    order='freq',
    levels=[6],
    stats=['mean', 'std', 'skew', 'kurtosis'],
    normalization=None,
):
    def get_statistics(coefficients, stats_list):
        features = []
        for stat in stats_list:
            if stat == 'mean':
                features.append(np.mean(coefficients))
            elif stat == 'std':
                features.append(np.std(coefficients))
            elif stat == 'skew':
                features.append(skew(coefficients))
            elif stat == 'kurtosis':
                features.append(kurtosis(coefficients))
        return features

    def extract_features_single_signal(signal, method, wavelet, level, order, levels, stats_list):
        wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=level)
        features = []

        for l in levels:
            coeffs =  wp.get_level(level, order)
            coeffs = np.array([c.data for c in coeffs])
            
            if method == "e+s":
                for c in coeffs:
                    stats = get_statistics(c, stats_list)
                    stats.append(np.sum(np.square(c)))
                    features.extend(stats)
                

            if method == 'energy' or method == 'min-max':
                 # Normalise
                coeffs = (coeffs - np.mean(coeffs)) / np.std(coeffs)
                
            if method == 'raw':
                features.extend(np.concatenate(coeffs))
            elif method == 'energy':       
                features.extend([np.sum(np.square(c)) for c in coeffs])
            elif method == 'entropy':
                features.extend([np.sum(entr(np.abs(c))) for c in coeffs])
            elif method == 'min-max':
                features.extend([np.min(c) for c in coeffs] + [np.max(c) for c in coeffs])
            elif method == 'stats':
                for c in coeffs:
                    features.extend(get_statistics(c, stats_list))
        
            # Optional normalisation for raw and energy
            if normalization == 'indvidual':
                features = (features - min(features)) / (max(features) - min(features))
                
        return features

    if combined:
        combined_signal = np.concatenate(signals, axis=0)
        combined_features = extract_features_single_signal(
            combined_signal, method, wavelet, level, order, levels, stats
            )
        features = combined_features
    else:
        features_list = [extract_features_single_signal(
            signal, method, wavelet, level, order, levels, stats
            ) for signal in signals]

        if combine_method == 'concatenate':
            features = np.concatenate(features_list)
        elif combine_method == 'interleave':
            features = np.ravel(np.column_stack(features_list))
        else:
            raise ValueError("Invalid combine_method. Choose from 'concatenate' or 'interleave'.")
    
    if normalization == 'combined':
        features = (features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0))
    
    return features
