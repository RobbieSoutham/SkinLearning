from argparse import ArgumentParser
import pandas as pd
from torch import nn
from ..NN.Models import DualDown, DualDownUp, DualUp, DualUpDown, MultiTemporal
from ..NN.Helpers import kfcv, DEVICE, set_seed
from ..Utils.Plotting import plot_parameter_bars, save_df
from ..Utils.Dataset import get_dataset
import seaborn as sns
import numpy as np
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
    models = [
        MultiTemporal(
            out="f_output",
            temporal_type="RNN",
            )
    ]
    
    names = [
        "RNN, Single FC, Final Output",
    ]

    dataset, scaler = get_dataset()

    accuracies = []
    p_accs = []
    maes = []
    for i, model in enumerate(models):
        print("--------------------------------------------------")
        print(f"Running model: {names[i]}) ({i+1}/{len(names)}")
        print("--------------------------------------------------")

        acc, p_acc, mae = kfcv(dataset, model, scaler, cluster=True)
        accuracies.append(acc)
        p_accs.append(p_acc)
        maes.append(mae)
        print("\n")

    # Build DF for results
    p_accs = np.array(p_accs)
    df = pd.DataFrame({
            "Architecture": names,
            "YM (Skin)": p_accs[:, 0],
            "YM (Adipose)": p_accs[:, 1],
            "PR (Skin)": p_accs[:, 2],
            "PR (Adipose)": p_accs[:, 3],
            "Perm (Skin)": p_accs[:, 4],
            "Perm (Adipose)": p_accs[:, 5],
            "Overall MAPE": accuracies,
            "Overall MAE": maes
        })
        
    df = df.set_index("Architecture")

    sns.set_theme()
    df.to_csv("Results/CNN_Temp.csv")
    save_df(df, fname="CNN_Temp")
    plot_parameter_bars(df, fname="CNN_Temp_bars")

def rnn_experiment():
    print()

def optimise():
    print()

if __name__ == "__main__":
    set_seed()

    if args.type == "CNN":
        cnn_experiment()
    elif args.type == "RNN":
        rnn_experiment()
    elif args.type == "Optimisation":
        optimise()

