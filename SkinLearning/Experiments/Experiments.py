from argparse import ArgumentParser
import pandas as pd
from torch import nn
from ..NN.Models import DualDown, DualDownUp, DualUp, DualUpDown
from ..NN.Helpers import KFCV
from ..Utils.Plotting import plotParameterBars, saveDf
from ..Utils.Dataset import getDataset

parser = ArgumentParser()

parser.add_argument(
    "-t", "--type",
    help="Type of experiemnts",
    type=str,
    choices = ["CNN", "RNN", "Optimisation"],
    # CNN tests feature extraction from just the convolutional layers
    # RNN tests the different types of outputs from RNN and derivatives
    # Optimisation optimised HPs,
    required=True
    )

parser.add_argument(
    "-m", "--model",
    nargs='+',
    help="The model/models for optimisation"
    )


args = parser.parse_args()

def cnn_experiment():
    # Test DualDownUp, DualUp, DualUp, DualDown, DualUpDown
    models = [
        #nn.DataParallel(DualDownUp()),
        nn.DataParallel(DualUp()),
        #nn.DataParallel(DualDown()),
        #nn.DataParallel(DualUpDown())

    ]
    
    names = [
        "Down Sample/Up Sample",
        "Up Sample",
        "Down Sample",
        "Up Sample/Down Sample"
    ]

    dataset, scaler = getDataset()

    accuracies = []
    p_accs = []
    for i, model in models:
        print("--------------------------------------------------")
        print(f"Running model: {names[i]}) ({i+1}/{len(names)}")
        print("--------------------------------------------------")

        acc, p_acc = KFCV(dataset, model, scaler, cluster=True)
        accuracies.append(acc)
        p_accs.append(p_acc)

    # Build DF for results
    df = pd.DataFrame({
            "Architecture": names,
            "YM (Skin)": p_accs[:, 0],
            "YM (Adipose)": p_accs[:, 1],
            "PR (Skin)": p_accs[:, 2],
            "PR (Adipose)": p_accs[:, 3],
            "Perm (Skin)": p_accs[:, 4],
            "Perm (Adipose)": p_accs[:, 5],
            "Overall": accuracies
        })
        
    df = df.set_index("Architecture")

    df.to_csv("../Results/FeatureExtraction.csv")
    saveDf(df, fname="FeatureExtractionDF")
    plotParameterBars(df, dfname="FeatureExtractionBars")

def rnn_experiment():
    print()

def optimise():
    print()

if __name__ == "__main__":
    if args.type == "CNN":
        cnn_experiment()
    elif args.type == "RNN":
        rnn_experiment()
    elif args.type == "Optimisation":
        optimise()

