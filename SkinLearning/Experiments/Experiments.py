from argparse import ArgumentParser
import itertools
import os
import pickle
import time
import pandas as pd
from SkinLearning.Utils.Misc import get_gpu_usage
from torch import nn
import torch
from SkinLearning.NN.Models import DualDown, DualDownUp, DualUp, DualUpDown, MultiTemporal
from SkinLearning.NN.Helpers import kfcv, DEVICE, set_seed
from SkinLearning.Utils.Plotting import plot_parameter_bars, save_df
from SkinLearning.Utils.Dataset import get_dataset
from .ModelArgs import *
import gc
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

seeds = [
    1235234,
    14336,
    431243,
    63453,
    273454,
    253462,
    25235,
    235235,
    2573982,
    2345
]
parser = ArgumentParser()

out_options = ['hidden']
temporal_types = ['LSTM', 'GRU', 'RNN']
single_fc_options = [True, False]
fusion_methods = ['independent', 'concatenate']

parser.add_argument(
    '-t', '--type',
    help='Type of experiemnts',
    type=str,
    choices = ['CNN', 'WPD', 'Optimisation', 'WPD_FC', 'Attention', 'Final'],
    # CNN tests feature extraction from just the convolutional layers
    # RNN tests the different types of outputs from RNN and derivatives
    # Optimisation optimised HPs,
    required=True
    )

parser.add_argument(
    '-te', '--temporal_type',
    nargs='+',
    help='The type of temporal net to use'
    )

parser.add_argument(
    '-f', '--fusion_method',
    type=str,
    help='The fusion method to use for WPD'
    )

parser.add_argument(
    '-m', '--model',
    type=int,
    help='Index of one of the top 3 models if required'
)

parser.add_argument(
    '-r', '--runs',
    nargs='+',
    help='List of runs for final comparison'
    )

parser.add_argument(
    '-s', '--single_fc',
    action='store_true',
    help='To use single or multiple FC layers'
    )

parser.add_argument(
    '-d', '--distributed',
    action='store_true',
    help='Whether distributed training is being used'
    )

args = parser.parse_args()

"""
    Returns model based on given parameters
"""
def init_model(**kwargs):

    model = MultiTemporal(**kwargs)
        
    # Use distributed training if set
    if args.distributed:
        """model.cuda(local_rank)
        model = DDP(model, device_ids=[local_rank])"""
        model = nn.DataParallel(model, device_ids=[0, 1])

    
    return model

"""
    Parses results for main KFCV experiments
"""
def get_top_results(limit_reached, results, start, fname):
    if limit_reached:
        # Save to continue later
        with open(f"Results/KFCV/{fname}_partial.pkl", "wb") as f:
            pickle.dump(results, f)
    else:

        # Sort the results by MAE in ascending order
        sorted_results = sorted(results, key=lambda x: x[0])

        columns=[
            'YM (Skin)',
            'YM (Adipose)',
            'PR (Skin)',
            'PR (Adipose)',
            'Perm (Skin)',
            'Perm (Adipose)',
            'Overall MAPE',
            'Overall MAE'
            ]
        top_df = pd.DataFrame(columns=columns)
        all_df = pd.DataFrame(columns=columns)

        # Print the top 3 models based on their MAE performance
        for i in range(len(sorted_results)):
            all_df.loc[str(sorted_results[i][-1])] = [
                    sorted_results[i][2][0],
                    sorted_results[i][2][1],
                    sorted_results[i][2][2],
                    sorted_results[i][2][3],
                    sorted_results[i][2][4],
                    sorted_results[i][2][5],
                    sorted_results[i][1],
                    sorted_results[i][0]
                    ]

            if i < 3:
                print(f'Model {i + 1}: {sorted_results[i][1]}, MAE: {sorted_results[i][0]}')
                top_df.loc[str(sorted_results[i][-1])] = [
                    sorted_results[i][2][0],
                    sorted_results[i][2][1],
                    sorted_results[i][2][2],
                    sorted_results[i][2][3],
                    sorted_results[i][2][4],
                    sorted_results[i][2][5],
                    sorted_results[i][1],
                    sorted_results[i][0]
                    ]


        top_df.to_csv(f'Results/KFCV/{fname}_top3.csv')
        all_df.to_csv(f'Results/KFCV/{fname}.csv')
        
        elapsed_time = time.time() - start
        hours = int(elapsed_time / 3600)
        minutes = int((elapsed_time % 3600) / 60)
        seconds = int(elapsed_time % 60)
        
        print(
            "Elapsed time: {} hours, {} minutes, {} seconds".format(
                hours, minutes, seconds
                )
            )

def cnn_temporal_sweep(temporal_type, single_fc):
    start = time.time()
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
    
    """combinations = list(itertools.product(
        out_options, single_fc_options
        ))"""
    dataset, scaler = get_dataset()

    results = []
    limit_reached = False
    for i, out in enumerate(out_options):
        # Check if time limit almost reached (71.2 hours)
        if time.time() - start >= 257400:
            
            limit_reached = True
            break

        print(
            f"Testing {temporal_type} with {'Single FC' if single_fc else 'Multi FC'} using {out}",
            f"{i+1}/{len(out_options)}"
            )

        model_args = {
            'conv': True,
            'out': out,
            'temporal_type': temporal_type,
            'single_fc': single_fc,
            'input_size': 15,
            'hidden_size': 128
        }

        mape, param_mape, mae = kfcv(
            dataset,
            scaler,
            model_init=init_model,
            model_args=model_args,
            cluster=False,
            track_str=f"for model {i+1}/{len(out_options)}",
            )
        results.append((mae, mape, param_mape, (out, temporal_type, single_fc)))
        
        print('\n')

    get_top_results(limit_reached, results, start, f"CNN/{temporal_type}_{'single' if single_fc else 'multi'}x")

def wpd_temporal_sweep(temporal_type, fusion_method=None, distributed=False):
    set_seed()
    combinations = list(itertools.product(
        out_options, single_fc_options
        ))

    extraction_args = {
            "signals": None,
            "method": "entropy",
            "combined": False,
            "wavelet": "db4",
            "level": 8,
            "order": "natural",
            "levels": [8],
            "normalization": False,
            "stats": None,
        }

    dataset, scaler = get_dataset(extraction_args=extraction_args)

    results = []
    start = time.time()
    limit_reached = False
    for i, (out, single_fc) in enumerate(combinations):
         # Check if time limit almost reached (71.5 hours)
        if time.time() - start >= 257400:
            
            limit_reached = True
            break

        print(
            f"Testing {temporal_type} with {fusion_method} and",\
                 f"{'Single FC' if single_fc else 'Multi FC'} using {out}",
            f"{i+1}/{len(combinations)}"
            )

        
        input_size = len(dataset[0]['input'])
        
        print(fusion_method)
        model_args = {
            'conv': False,
            'out': out,
            'temporal_type': temporal_type,
            'single_fc': single_fc,
            'input_size': input_size,
            'hidden_size': input_size*2,
            'fusion_method': fusion_method
        }

        mape, param_mape, mae = kfcv(
            dataset,
            scaler,
            model_init=init_model,
            model_args=model_args,
            cluster=False,
            track_str=f"for model {i+1}/{len(combinations)}",
            distributed=distributed
            )

        results.append(
            (mae, mape, param_mape, (out, temporal_type, fusion_method, single_fc))
            )
        
        #get_gpu_usage()

        get_top_results(limit_reached, results, start, f"WPD/{temporal_type}_{fusion_method}")

def best_wpd_fc_sweep():
    extraction_args = EXTRACTION_ARGS

    dataset, scaler = get_dataset(
        extraction_args=extraction_args,
        )

    model_args = [TOP_WPD_ARGS[0]]
            
    names = [WPD_NAMES[0]]

    
    run_experiment(model_args, names, f"WPD/Further FCs/top1_2fc", dataset, scaler)

def test_attention(temporal_type, model):
    models = []
    names = []

    if temporal_type == 'CNN':
        dataset, scaler = get_dataset()
        model_args = TOP_CNN_ARGS[model]
        names = CNN_NAMES[model]
    else:
        dataset, scaler = get_dataset(extraction_args=EXTRACTION_ARGS)
        model_args = TOP_WPD_ARGS[model]
        names = WPD_NAMES[model]
    
    model_args['attention'] = True
    
    run_experiment([model_args], [names], f"Attention/{temporal_type}_{model}_lowerPatience", dataset, scaler)


def multi_seed_comp(temporal_type, runs):
    names = []

    if temporal_type == 'CNN':
        dataset, scaler = get_dataset()
        model_args = TOP_CNN_ARGS[0]
        names = CNN_NAMES[0]
    else:
        dataset, scaler = get_dataset(extraction_args=EXTRACTION_ARGS)
        model_args = TOP_WPD_ARGS[0]
        names = WPD_NAMES[0]
    
    runs = [int(run) for run in runs]
    run_experiment(
        [model_args],
        [names],
        f"Final Comparison/{temporal_type}_{min(runs)}_{max(runs)}",
        dataset,
        scaler,
        runs=runs
        )


def run_experiment(model_args, names, fname, dataset, scaler, runs=None):
    if runs is None:
        runs = [0]

    all_df = None

    start_time = time.time()
    for i in runs:
        print(f'Starting kfold for run {i+1}/{len(runs)}')
        set_seed(seeds[i])

        accuracies = []
        p_accs = []
        maes = []
        for j, model_arg in enumerate(model_args):
            print('--------------------------------------------------')
            print(f'Running model: {names[j]} ({j+1}/{len(names)})')
            print('--------------------------------------------------')

            acc, p_acc, mae, train_losses, val_losses = kfcv(
                dataset,
                scaler,
                model_init=init_model,
                model_args=model_arg,
                track_str=f"for model {i+1}/{len(names)}")
            accuracies.append(acc)
            p_accs.append(p_acc)
            maes.append(mae)
            print('\n')

            # Record train/val curve for seed 1
            if i == 0:
                with open(f'Results/KFCV/{fname}_train_val.pkl', 'wb') as f:
                    pickle.dump([train_losses, val_losses], f)


        # Build DF for results
        p_accs = np.array(p_accs)
        cols = {
                'Architecture': names,
                'YM (Skin)': p_accs[:, 0],
                'YM (Adipose)': p_accs[:, 1],
                'PR (Skin)': p_accs[:, 2],
                'PR (Adipose)': p_accs[:, 3],
                'Perm (Skin)': p_accs[:, 4],
                'Perm (Adipose)': p_accs[:, 5],
                'Overall MAPE': accuracies,
                'Overall MAE': maes
        }

        if len(runs) != 1:
            cols['Run'] = [i for k in range(len(names))]

        df = pd.DataFrame(cols)
        
        if all_df is None:
            all_df = df
        else:
            all_df = pd.concat((all_df, df))
               
    all_df.to_csv(f'Results/KFCV/{fname}.csv')
    print(all_df)

    elapsed_time = time.time() - start_time
    hours = int(elapsed_time / 3600)
    minutes = int((elapsed_time % 3600) / 60)
    seconds = int(elapsed_time % 60)
    
    print(
        "Elapsed time: {} hours, {} minutes, {} seconds".format(
            hours, minutes, seconds
            )
        )

    with open(f'Results/KFCV/{fname}_time.txt', 'w') as f:
            f.write(str(elapsed_time))
def optimise():
    print()

"""# Get the global rank of the process from environment variables
rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])
world_size = int(os.environ['SLURM_NTASKS'])

# Initialize the distributed process group
torch.distributed.init_process_group(
    backend='nccl',
    init_method='env://'
)

# Set the current device to the local rank
torch.cuda.set_device(local_rank)"""



if __name__ == '__main__':
    set_seed()

    def choose_exp(temporal_type=None, fusion_method=None):
        if args.type == 'CNN':
            cnn_temporal_sweep(temporal_type, args.single_fc)
        elif args.type == 'WPD':
            wpd_temporal_sweep(temporal_type, fusion_method)
        elif args.type == 'Optimisation':
            optimise()
        elif args.type == 'WPD_FC':
            best_wpd_fc_sweep()
        elif args.type == 'Attention':
            test_attention(temporal_type, args.model)
        elif args.type == 'Final':
            multi_seed_comp(temporal_type, args.runs)

    # Iterate through selected models if given
    if args.temporal_type:
        for temporal_type in args.temporal_type:
            if args.fusion_method:
                choose_exp(temporal_type, args.fusion_method)
            else:
                choose_exp(temporal_type) 
    elif args.fusion_method:
        choose_exp(fusion_method=args.fusion_method) 
    else:
        choose_exp()

    

