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

parser.add_argument(
    '-t', '--type',
    help='Type of experiemnts',
    type=str,
    choices = ['CNN', 'WPD', 'Optimisation', 'WPD_FC', 'Final'],
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
    '-mg', '--multi_gpu',
    action='store_true',
    help='Whether to train using multiple GPUs in parallel'
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

def cnn_temporal_sweep(temporal_type):
    start = time.time()
    
    combinations = list(itertools.product(
        OUT_OPTIONS, SINGLE_FC_OPTIONS
        ))

    dataset, scaler = get_dataset()

    results = []
    limit_reached = False
    for i, (out, single_fc) in enumerate(combinations, temporal_type):
        print(
            f"Testing {temporal_type} with {'Single FC' if single_fc else 'Multi FC'} using {out}",
            f"{i+1}/{len(OUT_OPTIONS)}"
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
            track_str=f"for model {i+1}/{len(combinations)}",
            )
        results.append((mae, mape, param_mape, (out, temporal_type, single_fc)))
        
        print('\n')

    get_top_results(limit_reached, results, start, f"CNN/{temporal_type}_{'single' if single_fc else 'multi'}x")

def wpd_temporal_sweep(temporal_type, distributed=False):
    set_seed()

    combinations = list(itertools.product(
        OUT_OPTIONS, SINGLE_FC_OPTIONS, FUSION_METHODS
        ))

    dataset, scaler = get_dataset(extraction_args=EXTRACTION_ARGS)

    results = []
    start = time.time()
    limit_reached = False
    for i, (out, single_fc, fusion_method) in enumerate(combinations):
        print(
            f"Testing {temporal_type} with {fusion_method} and",\
                    f"{'Single FC' if single_fc else 'Multi FC'} {i+1/len(combinations)}"
            )

        
        input_size = len(dataset[0]['input'])
        
        print(fusion_method)
        model_args = {
            'conv': False,
            'out': out,
            'temporal_type': 'RNN',
            'single_fc': single_fc,
            'input_size': input_size,
            'hidden_size': input_size*2,
            'fusion_method': fusion_method
        }

        mape, param_mape, mae, _, _ = kfcv(
            dataset,
            scaler,
            model_init=init_model,
            model_args=model_args,
            cluster=False,
            track_str=f"for model {i+1}/{len(combinations)}",
            distributed=distributed
            )

        results.append(
            (mae, mape, param_mape, ('output', temporal_type, fusion_method, single_fc))
            )

    get_top_results(limit_reached, results, start, f"WPD/{temporal_type}_{'single' if single_fc else 'multi'}")

"""
    Test WPD on 1 or 3 FC layers to determine if further are necessary
"""
def best_wpd_fc_sweep():
    extraction_args = EXTRACTION_ARGS

    dataset, scaler = get_dataset(
        extraction_args=extraction_args,
        )

    model_args = [TOP_WPD_ARGS, TOP_WPD_ARGS],
    
    # Create a model for using 1 and 3 FC layers
    model_args[0]['single_fc'] = True
    model_args[0]['single_fc'] = False

            
    names = [WPD_NAME, WPD_NAME]
    names[0][-1] = 'FC x1'
    names[1][-1] = 'FC x2'

    
    run_experiment(model_args, names, f"WPD/Stats_entr/stats_single", dataset, scaler)

"""
    Run KFCV with a different seed for each run
"""
def multi_seed_comp(temporal_type, runs):
    names = []

    if temporal_type == 'CNN':
        dataset, scaler = get_dataset()
        model_args = TOP_CNN_ARGS[CNN_BEST_IDX]
        names = CNN_NAMES[CNN_BEST_IDX]
        input_size = len(dataset[0]['input'])
        print(input_size)
    else:
        dataset = torch.load('Data/WPD DS/dataset_stats.pt')
        with open('Data/WPD DS/scaler_stats.pkl', 'rb') as f:
            scaler = pickle.load(f)
        model_args = TOP_WPD_ARGS
        names = WPD_NAME
        input_size = len(dataset[0]['input'])
        print(input_size)
        model_args['input_size'] = input_size
        model_args['hidden_size'] = input_size*2
    
    runs = [int(run) for run in runs]
    run_experiment(
        [model_args],
        [names],
        f"/Test/{temporal_type}/{temporal_type}_L1_{min(runs)}_{max(runs)}",
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

def optimise(temporal_type, model=None):
    set_seed()
    if temporal_type == 'CNN':
        dataset, scaler = get_dataset()
        model_args = TOP_CNN_ARGS[model]
        names = CNN_NAMES[model]
    else:
        dataset, scaler = get_dataset(extraction_args=EXTRACTION_ARGS)
        model_args = TOP_WPD_ARGS
        names = WPD_NAME

    # Test only lower batch_sizes for generalisation
    # Increasing likely not to lead to improvement
    batch_sizes = [8, 16, 32]

    loss_fns = [nn.MSELoss(), nn.L1Loss()]

    best_params = []
    best_loss = 10e3

    total_exps = len(loss_fns) * len(batch_sizes)
    i = 0
    for loss_fn in loss_fns:
        for batch_size in batch_sizes:
            print(
            f'Testing {temporal_type} {names} with {batch_size} batch size and {loss_fn} {i+1}/{total_exps}'
            )

            mape, param_mape, mae, train_losses, val_losses = kfcv(
                dataset,
                scaler,
                model_init=init_model,
                model_args=model_args,
                cluster=False,
                track_str=f"for model {i+1}/{total_exps}",
                criterion=loss_fn,
                batch_size=batch_size
                )

            if best_loss < mae:
                best_res = [mae, mape, param_mape, train_losses, val_losses]
                best_params = [batch_size, loss_fn]
                best_loss = mae
            i += 1

    param_loss = best_res[2]

    cols = {
            'YM (Skin)': param_loss[0],
            'YM (Adipose)': param_loss[1],
            'PR (Skin)': param_loss[2],
            'PR (Adipose)': param_loss[3],
            'Perm (Skin)': param_loss[4],
            'Perm (Adipose)': param_loss[5],
            'Overall MAPE': best_res[1],
            'Overall MAE': best_res[0],
            'Temporal type': names[1],
            'Out': names[0],
            'Loss fn': best_params[1],
            'Batch size': best_params[0]
    }
    

    # onvert to single element list
    if model is not None:
        for key in cols.keys():
            cols[key] = [cols[key]]
    
    if temporal_type == 'WPD':
        cols['Fusion Method'] = names[-2]
        cols['FC'] = names[3]
    else:
        if names[-1]:
            cols['FC'] = 'FC x1'
        else:
            cols['FC'] = 'FC x3'

    best_df = pd.DataFrame(cols)
    print(best_df)

    best_df.to_csv(f'Results/KFCV/Optimization/{temporal_type}_{model}_test')
    with open(f'Results/KFCV/Optimization/{temporal_type}_{model}_train_val_test.pkl', 'wb') as f:
        pickle.dump([best_res[-2], best_res[-1]], f)
        

if __name__ == '__main__':
    set_seed()

    def choose_exp(temporal_type=None, fusion_method=None):
        if args.type == 'CNN':
            cnn_temporal_sweep(temporal_type, args.single_fc)
        elif args.type == 'WPD':
            wpd_temporal_sweep(temporal_type, fusion_method, single_fc=args.single_fc)
        elif args.type == 'Optimisation':
            optimise(temporal_type, args.model)
        elif args.type == 'WPD_FC':
            best_wpd_fc_sweep()
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

    

