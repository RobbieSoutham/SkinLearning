import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(train_loader, net, LR=0.1, epochs=2000, val_loader=None):
    net.to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    criterion = nn.L1Loss()
    val_losses = []        
    losses = []
    last_loss = 0

    print(f"Using: {DEVICE}")
    
    for epoch in range(epochs):
        loss = 0
        net.train()
        with tqdm(train_loader, unit="batch") as it:
            if epoch > 0:
                it.set_postfix(lastLoss=last_loss, valLoss=0 if len(val_losses) \
                     == 0 else val_losses[-1])
            for idx, data in enumerate(it):
                it.set_description(f"Epoch {epoch+1}/{epochs}")
                inp, out = data['input'].to(DEVICE), data['output'].to(DEVICE)
                
                optimizer.zero_grad()
                predicted = net(inp)

                cost = criterion(out, predicted)
                loss += cost.item()
                cost.backward()
                optimizer.step()
        
        if val_loader:
            val_loss = 0
            net.eval()
            for idx, data in enumerate(val_loader):
                inp, out = data['input'].to(DEVICE), data['output'].to(DEVICE)

                predicted = net(inp)
                cost = criterion(out, predicted)
                val_loss += cost.item()
            val_loss /= len(val_loader)  
            val_losses.append(val_loss)
        
        loss /= len(it)
        losses.append(loss)
        last_loss = loss
         
    print(f"Average train loss: {np.sum(losses)/epochs}")
    print(f"Average validation loss: {np.sum(val_losses)/epochs}")
    
    return losses, val_losses

"""
    Validates the given model on the given validation loader
    Returns 100-MAPE for each batch and each parameter
"""
def test(test_loader, net, scaler):
    net.to(DEVICE)
    net.eval()
    criterion = nn.L1Loss()

    losses = []
    p_losses = []
    mae = []

    with torch.no_grad():
        with tqdm(test_loader, unit=" batch") as it:
            for idx, data in enumerate(it):
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
def getParameterLoss(models, names, test_loader, print=False):
    params = []
    overall = []
    
    # Run evaluation on all models
    for model in models:
        ps, avg, _ = test(test_loader, model)
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