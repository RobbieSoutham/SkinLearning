from copy import deepcopy
import torch
from torch import nn

""" Models for testing outputs and RNN derivatives """

""" Best CNN found for feature extraction """
best_CNN = nn.Sequential(
    nn.Conv1d(2, 128, kernel_size=5, padding=1, bias=False),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=5, stride=2),

    nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=2, stride=2),

    nn.Conv1d(256, 512, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=2, stride=2),
)

"""
    Configurable CNN + Temporal network
    
    Parameters:
        conv: (boolean):
            If true uses the "best_CNN" otherwise just the temporal network.
        input_size (int):
            Size of the input passed into the temporal network.
        hidden_size (int):
            Number of hidden layers in the temporal network.
        layers (int):
            Number of layers in the temporal network.
        single_fc (boolean):
            If true only one FC layer is used, otherwise three are used.
        out (string):
            Name of the output to use.
            Options:
                f_hidden: final hidden state,
                hidden: full hidden state,
                f_output: final output,
                output: full output,
                h+o: combination of final hidden state and final output.
        temporal_type (string):
            Type of model to use (RNN, LSTM, GRU)
"""
class MultiTemporal(nn.Module):
    def __init__(
        self,
        conv=True,
        input_size=15,
        hidden_size=256,
        single_fc=True,
        out="f_hidden",
        layers=1,
        temporal_type="RNN",
        fusion_method="concatenate"
    ):
        super(MultiTemporal, self).__init__()

        self.hidden_size = hidden_size
        self.out = out
        self.temporal_type = temporal_type
        self.fusion_method = fusion_method
        self.input_size = input_size
        self.conv = conv
        
        if conv:
            self.cnn = deepcopy(best_CNN)
        

        if temporal_type == "RNN":
            net = nn.RNN
        elif temporal_type == "LSTM":
            net = nn.LSTM
        elif temporal_type == "GRU":
            net = nn.GRU
        else:
            raise Exception("Not a valid NN type.")

        if fusion_method == 'concatenate':
            self.net = net(input_size, hidden_size, layers, batch_first=True)
        elif fusion_method == 'multi_channel':
            self.net = net(2, hidden_size, layers, batch_first=True)
        elif fusion_method == 'independent':
            self.net = net(input_size//2, hidden_size, layers, batch_first=True)
        else:
            raise ValueError("Invalid method. Choose from 'concatenate', 'multi_channel', or 'independent'.")
        
        # Check size of output to determine FC input
        input_tensor = torch.zeros(
            32,
            512,
            input_size if fusion_method == "concatenate" else input_size//2
        )
        output, _ = self.net(input_tensor)
        
        fc_in = hidden_size
        if out == 'output':
            fc_in = output.shape[1] * output.shape[2]
        elif out == 'f-output':
            fc_in = output.shape[2]
        elif fc_in == 'hidden' or out == 'cell':
            out = hidden_size * output.shape[2]
        elif out == 'f-hiden' or out == 'f-cell':
            fc_in = output.shape[2]
        elif out == 'h+o' or out == 'h+c' :
            fc_in = output.shape[1]
        
        if single_fc:
            self.fc = nn.Linear(fc_in*layers, 6)
        else:
            self.fc = nn.Sequential(
                nn.Linear(fc_in*layers, 128),
                nn.ReLU(),
                nn.Linear(128 , 64),
                nn.ReLU(),
                nn.Linear(64, 6),   
            )

    def forward(self, x):
        batch_size = x.shape[0]
        
        if self.conv:
            x = self.cnn(x)
        else:
            x = x.reshape(batch_size, -1, self.input_size)

        def getOutputs(x):
            if self.temporal_type == "LSTM":
                o, (h, _) = self.net(x)
            else:
                o, h = self.net(x)
            
            return o, h

        
        if self.fusion_method == 'multi_channel':
            o, h = getOutputs(x.view(batch_size, -1, 2))
        elif self.fusion_method == 'independent':
            signal_size = self.input_size//2
            signal1 = x[..., :signal_size].reshape(batch_size, -1, signal_size)
            signal2 = x[..., signal_size:].reshape(batch_size, -1, signal_size)

            o1, h1, = self.net(signal1)
            o2, h2 = self.net(signal2)
        else:
            o, h = getOutputs(x)
            
        
        if self.temporal_type == "LSTM":
            o, (h, _) = self.net(x)
        else:
            o, h = self.net(x)
        
        if self.out == "f_hidden":
            if self.fusion_method == "independent":
                x = torch.concat(
                    [h1[-1], h2[-1]], dim=1
                    ).reshape(batch_size, -1)
            else:
                x = h[-1].reshape(batch_size, -1)
            x = h[-1].reshape(batch_size, -1)
        elif self.out == "hidden":
            x = h.reshape(batch_size, -1)
        elif self.out == "f_output":
            if self.fusion_method == "independent":
                x = torch.concat(
                    [o1[:, -1, :], o2[:, -1, :]], dim=1
                    ).reshape(batch_size, -1)
            else:
                x = o[:, -1, :].reshape(batch_size, -1)
        elif self.out == "output":
            x = o.reshape(batch_size, -1)
        elif self.out == "h+o":
            x = torch.concat([h[-1], o[:, -1, :]], dim=1).view(o.size(0), -1)
            
        x = self.fc(x)
        return x

""" Models for feature extraction testing """

"""
    Up samples to 128 from 32
"""
class DualDownUp(nn.Module):
    def __init__(self):
        super(DualDownUp, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=1),
    
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(32, 6)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)

        return x

# Up samples to 128 from 32
class DualUpDown(nn.Module):
    def __init__(self):
        super(DualUpDown, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 128, kernel_size=5, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        
        self.fc = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(16, 6)
        )

def forward(self, x):
    batch_size = x.shape[0]
    x = self.cnn(x)

    x = x.view(batch_size, -1)

    x = self.fc(x)
    
    return x

"""
    Up samples to 128 from 32
"""
class DualDownUp(nn.Module):
    def __init__(self):
        super(DualDownUp, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),     
        )

        self.fc = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(32, 6)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.cnn(x)
        x = self.fc(x)

        return x

"""
    Up samples to 128 from 32
"""
class DualDownUp(nn.Module):
    def __init__(self):
        super(DualDownUp, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(32, 6)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.cnn(x)
        

        x = x.view(batch_size, -1)

        x = self.fc(x)
        return x

"""
    Up samples to 256 from 128
"""
class DualUp(nn.Module):
    def __init__(self):
        super(DualUp, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(2, 128, kernel_size=5, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),

            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(7680, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 6)
        )


    def forward(self, x):
        batch_size = x.shape[0]
        x = self.cnn(x)
        
        x = x.view(batch_size, -1)
        
        x = self.fc(x)
        
        x = self.fc4(x)
        return x

"""
    Down samples from 128 features
"""
class DualDown(nn.Module):
    def __init__(self):
        super(DualDown, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(2, 128, kernel_size=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),

            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),

            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),

            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),
        )

        self.fc = nn.Sequential(
            nn.Linear(112, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(16, 6)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.cnn(x)
        
        x = x.view(batch_size, -1)

        x = self.fc(x)

        return x

"""
    Improvements on CNN
"""