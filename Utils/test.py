import os
import pandas as pd
import pickle

if __name__ == "__main__":
     #print(os.listdir("SamplingResults2/"))
     incomplete = []
     for run in os.listdir("SamplingResults2/"):
          if len(os.listdir(f"SamplingResults2/{run}"))!= 2:
              incomplete.append(run)
          else:
               for file in os.listdir(f"SamplingResults2/{run}"):
                    if pd.read_csv(f"SamplingResults2/{run}/{file}")['0'].max() != 7.0:
                         incomplete.append(run)
     
     with open("missingSamples.pkl", "wb") as f:
          pickle.dump(incomplete, f)
     
     print("Complete,", len(incomplete), "missing")

