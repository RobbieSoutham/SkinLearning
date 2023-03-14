import pickle
if __name__ == "__main__":
    with open("missingSamples.pkl", "rb") as f:
        print(pickle.load(f))