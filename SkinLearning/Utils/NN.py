import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(train_loader, net, LR=0.1, epochs=2000, val_loader=None):
    net.to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    criterion = nn.L1Loss()
    all_MSE = nn.L1Loss()
    val_losses = []
    print(f"Using: {DEVICE}")
                            
    parameter_loss = []
    losses = []
    processed = 0
    last_loss = 0
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
                
                for i in range(len(predicted)):
                    current_MSE = []
                    for j in range(6):
                        current_MSE.append(all_MSE(out[i][j], predicted[i][j]).item())
                    parameter_loss.append(current_MSE)
                    processed += 1
        
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
    
    print("Parameters: Skin YM, Adipose YM, Skin PR, Adipose PR, Skin Perm, Adipose Perm")
    print(f"Sampled Ranges: 10e3 - 50e3, 1e3 - 25e3, 0.48 - 0.499, 0.48 - 0.499, 10e - 12-10e10, 10e-12 - 10e10") 
    print(f"Average parameter loss: {np.mean(np.reshape(np.array(parameter_loss), (-1, 6)), axis=0)}")        
    print(f"Average overall loss: {np.sum(losses)/processed}")
    
    return losses, val_losses

"""
    Validates the given model on the given validation loader
    Returns 100-MAPE for each batch and each parameter
"""
def test(test_loader, net, scaler):
    net.to(DEVICE)
    net.eval()
    criterion = nn.L1Loss()
    crit = nn.L1Loss()
    differences = []
    losses = []
    p_losses = []

    with torch.no_grad():
        with tqdm(test_loader, unit=" batch") as it:
            for idx, data in enumerate(it):
                inp, out = data['input'].to(DEVICE), data['output'].to(DEVICE)

                predicted = net(inp)

                # Get overall MAE
                cost = criterion(predicted, out)
                loss = cost.item()
                
                # Convert to 100-MAPE
                losses.append(100-(100*loss))

                p_loss = []
                # Loop over each parameter
                for i in range(6):
                    p = predicted[:, i]
                    o = out[:, i]

                    # Get MAE
                    cost = criterion(p, o)
                    loss = cost.item()

                    # Convert to 100-MAPE
                    p_loss.append(100-(100*loss))
                    
                    p = p.cpu().numpy()
                    o = o.cpu().numpy()


                    for i in range(len(predicted)):

                        p = predicted[i].cpu().numpy().reshape(1, -1)
                        o = out[i].cpu().numpy().reshape(1, -1)

                        p = scaler.inverse_transform(p)[0]
                        o = scaler.inverse_transform(o)[0]

                        differences.append(100-(np.abs(p-o)/o)*100)
            p_losses.append(p_loss)
            
    return np.mean(p_losses, axis=0), np.mean(losses, axis=0), np.mean(differences, axis=0) #np.mean(np.array(differences), axis=0)