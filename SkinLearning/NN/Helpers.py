import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
    Sets a given seed globally.

    Parameters:
        seed (int):
            The seed to be used.
"""
def set_seed(seed=123):
    random.seed(seed) # Python rnadom number generator, used by sklearn
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def update_plot(losses, val_losses, ax, label, epoch):
    # Clear the current plot
    ax.clear()

    # Add the new data point to the existing plot
    ax.plot(range(epoch+1), losses, label="Training")
    ax.plot(range(epoch+1), val_losses, label="Validation")

    ax.legend()
    ax.figure.canvas.draw()
   

"""
    Trains a given model with a given validation training set.
    Validates after every epoch if validation loader is given.

    Parameters:
        tain_loader (torch.utils.data.DataLoader):
            Torch train loader for the dataset.
        net (torch.nn):
            The model to train.
        LR (float):
            The learning rate for the optimiser.
        val_loader (torch.utils.data.DataLoader):
            If given will validate after every epoch.
        early_stopping (boolean):
            If true early stopping will be used.
        optimizer (torch.optim):
            The optimizer to use for training.
        plot (boolean):
            If true will dynamically plot training and validation loss.
            Note: need to use "%matplotlib inline" in the notebook.
"""
def train(
    train_loader,
    net,
    LR=0.1,
    epochs=2000,
    val_loader=None,
    early_stopping=False,
    patience=50,
    optimizer=optim.Adam,
    plot=False,
    cluster=False
):
    net.to(DEVICE)
    optimizer = optimizer(net.parameters(), lr=LR)
    criterion = nn.L1Loss()
    val_losses = []        
    losses = []
    best_val_loss = 1e10
    counter = 0
    
    global loss
    loss = 0
    
    if plot:
        _, ax = plt.subplots(1, 1)

    print(f"Using: {DEVICE}")

    def processBatch(ittr):
        global loss
        loss = 0
        
        for _, data in enumerate(ittr):
            inp, out = data['input'].to(DEVICE), data['output'].to(DEVICE)

            optimizer.zero_grad()
            predicted = net(inp)

            cost = criterion(out, predicted)
            loss += cost.item()
            cost.backward()
            optimizer.step()
    
    for epoch in range(epochs):
        net.train()
        
        if plot or cluster:
            processBatch(train_loader)
        else:
            with tqdm(train_loader, unit="batch") as it:
                if epoch > 0:
                    it.set_postfix(lastLoss=losses[-1], valLoss=0 if len(val_losses) \
                         == 0 else val_losses[-1], counter=counter, epoch=epoch+1/epochs)
                processBatch(it)
        
        loss /= len(train_loader)
        losses.append(loss)

        if val_loader:
            val_loss = 0
            net.eval()
            for _, data in enumerate(val_loader):
                inp, out = data['input'].to(DEVICE), data['output'].to(DEVICE)

                predicted = net(inp)
                cost = criterion(out, predicted)
                val_loss += cost.item()
            val_loss /= len(val_loader)  
            val_losses.append(val_loss)
            
            if plot:
                update_plot(losses, val_losses, ax, "Training", epoch)
            if cluster:
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"    Training loss: {losses[-1]}")
                print(f"    Validation loss: {val_losses[-1]}")
                print(f"    Stagnation counter: {counter}\n")

            if early_stopping:
                if val_losses[-1] < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                if counter >= patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break
         
    print(f"Average train loss: {np.sum(losses)/epochs}")
    print(f"Average validation loss: {np.sum(val_losses)/epochs}")
    
    return losses, val_losses

"""
    Validates the given model on the given validation loader
    Returns 100-MAPE for each batch and each parameter.

    Parameters:
        test_loader (torch.utils.data.DataLoader):
            The test loader to use in the forward pass.
        net (torch.nn):
            The model to test
        scaler (sklearn.preprocessing.MinMaxScaler):
            The scaler used when creating the dataset.
            This is needed to denormalise and find 100-MAPE.
    
"""
def test(test_loader, net, scaler, cluster=True):
    net.to(DEVICE)
    net.eval()
    criterion = nn.L1Loss()
    losses = []
    p_losses = []
    mae = []

    def testBatch(it):
        for _, data in enumerate(it):
                inp, out = data['input'].to(DEVICE), data['output'].to(DEVICE)
                predicted = net(inp)
                
                # Denormalise
                p = scaler.inverse_transform(predicted.cpu().numpy())
                o = scaler.inverse_transform(out.cpu().numpy())
                    
                # Get column wise and overall MAPE
                # Since each column is normalised should also be able to use MAE*100
                p_loss = np.mean(100*(np.abs(o-p)/o), axis=0)
                loss = np.mean(100*(np.abs(o-p)/o))

                mae.append(criterion(predicted, out).item())
            
                p_losses.append(p_loss)
                losses.append(loss)

    with torch.no_grad():
        if not cluster:
            with tqdm(test_loader, unit=" batch") as it:
                testBatch(it)
        else:
            testBatch(test_loader)
            
    
            
    average_mape = 100 - np.mean(losses)
    average_p_loss = 100 - np.mean(p_losses, axis=0)
    mae_mean = np.mean(mae)
    
    return average_mape, average_p_loss, mae_mean

"""
    Run evaluation and build a dataframe of parameter accuracies
    Accuracies are calculated as 100-MAPE
    
    Parameters:
        models (list): The networks to test
        names (list): Names of the models to label the dataframe
        test_loader (DataLoader): The dataloader for the testing set
"""
def get_parameter_loss(models, names, test_loader, scaler, print=False):
    params = []
    overall = []
    
    # Run evaluation on all models
    for model in models:
        ps, avg, _, _ = test(test_loader, model, scaler)
        overall.append(avg)
        params.append(ps)
    
    all_vals = np.array(params)
    df = pd.DataFrame({
        "Architecture": names,
        "YM (Skin)": all_vals[:, 0],
        "YM (Adipose)": all_vals[:, 1],
        "PR (Skin)": all_vals[:, 2],
        "PR (Adipose)": all_vals[:, 3],
        "Perm (Skin)": all_vals[:, 4],
        "Perm (Adipose)": all_vals[:, 5],
        "Overall": overall
    })
    
    df = df.set_index("Architecture")

    if print:
        df = df.style.set_caption(
            'Average percent correctness 100-MAPE').set_table_styles([{
            'selector': 'caption',
            'props': [
                ('color', 'black'),
                ('font-size', '16px'),
                ('text-align', 'center')
            ]
        }])

        display(df)
    else:
        return df


def kfcv(dataset, model, scaler, k=5, cluster=False):
    # Initialize k and KFold
    kfold = KFold(n_splits=k, shuffle=True)

    # Perform k-fold cross-validation
    accuracies = []
    p_accuracies = []

    for fold, (train_index, valid_index) in enumerate(kfold.split(dataset), start=1):
        print(f"Testing fold {fold}")
        
        train_set = Subset(dataset, train_index)
        valid_set = Subset(dataset, valid_index)

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=32)

        # Train the model
        train(
            train_loader,
            model,
            val_loader=valid_loader,
            LR=0.0001,
            epochs=300,
            cluster=cluster)

        accuracy, p_accuracy, _ = test(valid_loader, model, scaler, cluster=cluster)
        accuracies.append(accuracy)
        p_accuracies.append(p_accuracy)
        print(f"Fold {fold} accuracy: {accuracy:.2f}%")

    # Calculate average accuracy across all folds
    avg_accuracy = np.mean(accuracies)
    avg_p_accuracy = np.mean(p_accuracies, axis=0)
    print(f"Average accuracy: {avg_accuracy:.2f}%")

    return avg_accuracy, avg_p_accuracy
