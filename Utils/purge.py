import os

if __name__ == "__main__":
    os.system("cd SamplingResults/")
    subfolders = [ f.path for f in os.scandir("") if f.is_dir() ]

    for folder in subfolders:
        if "Disp2.csv" in os.listdir():
            print("Found")