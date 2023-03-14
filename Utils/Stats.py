from sklearn.metrics import mean_squared_error
import random
import pickle
import os
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

def get():
    me = []
    mse = []
    sse = []
    pts = list(range(25, 500, 25))

    f = open(f"{SAMPLE_FILE}", "rb")
    samples = pickle.load(f)
    f.close()
    for steps in pts:
        print(f"Running for step size: {steps}")
        x_interped = []
        x_origin = []

        exps = os.listdir(f"{SIGNAL_FOLDER}/")
        fail = False
        
        for run in exps:
            print(f"   Run: {run}")
            inp = []
            fail = False
            files = os.listdir(f"{SIGNAL_FOLDER}/{run}/")

            if files != ['Disp1.csv', 'Disp2.csv']:
                continue
            
            processed = 0
            for file in files:
                a = pd.read_csv(f"{SIGNAL_FOLDER}/{run}/{file}")
                a.rename(columns = {'0':'x', '0.1': 'y'}, inplace = True)

                # Interpolate curve for consistent x values
                xNew = np.linspace(0, 7, num=steps, endpoint=True)
                interped = interp1d(a['x'], a['y'], kind='cubic', fill_value="extrapolate")(xNew)
                if a['x'].max() != 7.0:
                    fail = True
                    print(run, "has failed")
                    
                    break

                # Find closest value in original to interpreted
                orig = np.array(a['y'].values)
                closest = []
                
                for y in interped:
                    diff = []
                    for y_2 in orig:
                        # Get difference
                        diff.append(abs(y_2 - y))

                    # Find location of smallest difference
                    idx = np.array(diff).argmin()
                    # Use as representative point
                    closest.append(orig[idx])

                x_interped.append(interped.astype(np.float32))
                x_origin.append(closest)
                processed += 1

            if fail and processed == 1:
                del x_interped[-1]
                del x_origin[-1]
        
        # Get metrics
        x_origin = np.array(x_origin).flatten()
        x_interped = np.array(x_interped).flatten()
        mse.append(mean_squared_error(x_origin, x_interped))
        sse.append(mse[-1]*len(x_origin))
        me.append(np.max(np.absolute(np.array(x_origin) - np.array(x_interped))))
   
    f =  open(f"statsRes.pkl", "wb")
    pickle.dump([mse, sse, me], f)
    f.close()
        
    print(f"{mse}\n\n {sse}\n\n {me}")

SAMPLE_FILE = "filtered.pkl"
SIGNAL_FOLDER = "SamplingResults2"

if __name__ == "__main__":
    get()