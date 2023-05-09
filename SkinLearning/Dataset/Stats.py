from sklearn.metrics import mean_squared_error
import random
import pickle
import os
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

SAMPLE_FILE = "samples.pkl"
SIGNAL_FOLDER = "SamplingResults"

"""
    Obtains the interpolation error for a number of frequencies.
"""
if __name__ == "__main__":
    me = []
    mse = []
    mae = []
    sse = []
    pts = range(50, 500, 50)

    with open(f"Data/newSamples.pkl", "rb") as f:
        samples = pickle.load(f).astype(np.float32)
        for steps in pts:
            x_interped = []
            x_origin = []
            
            current_me = []
            current_mse = []
            current_mae = []
            current_sse = []

            runs = 0
            exps = os.listdir(f"D:/SamplingResults2/")
            exps = [int(exp) for exp in exps]

            for run in tqdm(exps):
                inp = []
                fail = False

                files = os.listdir(f"D:/SamplingResults2/{run}/")

                if files != ['Disp1.csv', 'Disp2.csv']:
                    continue

                for file in files:
                    df = pd.read_csv(f"D:/SamplingResults2/{run}/{file}")
                    df.rename(columns = {'0':'x', '0.1': 'y'}, inplace = True)
                    if df['x'].max() != 7.0:
                        failed = True
                        break
    
                    xs, ys = df['x'], df['y']
                        # Interpolate curve for consistent x values
                    xNew = np.linspace(0, 700, num=steps, endpoint=False)
                    interped = interp1d(xs, ys, kind='cubic', fill_value="extrapolate")(xNew)

                    # Create an interpolation function for the original curve
                    f_original = interp1d(np.arange(len(ys)), ys, bounds_error=False)

                    # Interpolate the original curve using the same method as the interpolated curve
                    interpolated_original = interped.copy()
                    for i in range(len(interped)):
                        interpolated_original[i] = f_original(i)

                    interpolated_original = np.nan_to_num(interpolated_original, 1e100)

                    current_mse.append(mean_squared_error(interpolated_original, interped))
                    current_me.append(np.max(np.abs(interpolated_original - interped)))
                    current_mae.append(np.mean(np.abs(interpolated_original - interped)))
                    current_sse.append(np.sum(current_mse[-1] * len(interped)))
                
            mse.append(np.mean(current_mse))
            me.append(np.mean(current_me))
            mae.append(np.mean(current_mae))
            sse.append(np.mean(current_sse))


            with open(f"statsRes.pkl", "wb") as f:
                pickle.dump([mse, sse, me, steps], f)
