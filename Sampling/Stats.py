from sklearn.metrics import mean_squared_error
import random
import pickle
import os
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d

SAMPLE_FILE = "samples.pkl"
SIGNAL_FOLDER = "SamplingResults"

if __name__ == "__main__":
    me = []
    mse = []
    sse = []
    pts = range(1, 500, 25)

    with open(f"{SAMPLE_FILE}", "rb") as f:
        samples = pickle.load(f).astype(np.float32)
        print(len(pts))
        for steps in pts:
            x_interped = []
            x_origin = []
            
            runs = 0
            exps = os.listdir(f"{SIGNAL_FOLDER}/")
            exps = [int(exp) for exp in exps]
            random.shuffle(exps)
            exps = exps[:19999]
            print()
            
            for run in tqdm(exps[:19999]):
                inp = []
                fail = False

                files = os.listdir(f"{SIGNAL_FOLDER}/{run}/")

                if files != ['Disp1.csv', 'Disp2.csv']:
                    continue
                    
                

                for file in files:
                    a = pd.read_csv(f"{SIGNAL_FOLDER}/{run}/{file}")
                    a.rename(columns = {'0':'x', '0.1': 'y'}, inplace = True)
                    #a = pd.concat([pd.DataFrame([[0,0.1]], columns=a.columns), a], ignore_index=True)


                    # Interpolate curve for consistent x values
                    xNew = np.linspace(0, 7, num=steps, endpoint=False)
                    interped = interp1d(a['x'], a['y'], kind='cubic', fill_value="extrapolate")(xNew)
                    
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


                    #fix, ax = plt.subplots(1, 2)
                    #ax[0].plot(xNew, interped)
                    #a.plot(ax=ax[1], x='x', y='y')
                    #if run == 3:
                       # break


                    #if len(a) < 702:
                    #    print(f"{signalFolder}/{run}/{file}: {len(a)}")
                   #     fail = True
                     #   break

                   # while len(a) > 702:
                    #    a = a.drop(index=np.random.randint(0, len(a)-1)).reset_index(drop=True)

                    #print(a)

                    x_interped.append(interped.astype(np.float32))
                    x_origin.append(closest)
                    #print(closest, "\n\n", x_origin)
                    
                    


            if not fail:
                #print(x_origin, "\n", x_interped, "\n", steps)
                mse.append(mean_squared_error(x_origin, x_interped))
                sse.append(mse[-1]*len(x_origin))
                me.append(np.max(np.absolute(np.array(x_origin) - np.array(x_interped))))

        with open(f"statsRes.pkl", "wb") as f:
            pickle.dump([mse, sse, me, steps], f)
