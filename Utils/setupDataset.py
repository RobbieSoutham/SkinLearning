from SkinLearning.Utils.Dataset import get_dataset
from SkinLearning.Experiments.ModelArgs import EXTRACTION_ARGS, EXTRACTION_ARGS_STATS
import torch
import pickle

if __name__ == '__main__':
    dataset, scaler = get_dataset(extraction_args=EXTRACTION_ARGS_STATS)

    torch.save(dataset, 'Data/WPD DS/dataset.pt')

    with open('Data/WPD DS/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

