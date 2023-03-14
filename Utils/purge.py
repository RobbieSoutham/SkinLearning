import os

if __name__ == "__main__":
    subfolders = os.listdir("SamplingResults/")

    for folder in subfolders:
        files = os.listdir(f"SamplingResults/{folder}")
        if "Disp2.csv" in files:
            for file in files:
                    if file != "Disp1.csv" and file != "Disp2.csv":
                              os.remove(f"SamplingResults/{folder}/{file}")
       
