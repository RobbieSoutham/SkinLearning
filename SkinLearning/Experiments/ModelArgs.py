TOP_WPD_ARGS = {
    'conv': False,
    'out': 'output',
    'temporal_type': 'LSTM',
    'single_fc': True,
    'input_size': 512,
    'hidden_size': 1024,
    'fusion_method': 'independent'
}

TOP_CNN_ARGS = [
        {
            'out': 'output',
            'temporal_type': 'LSTM',
            'single_fc': True,
            'input_size': 15,
            'hidden_size': 128
        },
        {
            'out': 'output',
            'temporal_type': 'GRU',
            'single_fc': False,
            'input_size': 15,
            'hidden_size': 128,
        },
        {
            'out': 'output',
            'temporal_type': 'LSTM',
            'single_fc': True,
            'input_size': 15,
            'hidden_size': 128,
        }
]

EXTRACTION_ARGS_STATS = {
            "signals": None,
            "method": "stats",
            "combined": False,
            "wavelet": "db4",
            "level": 8,
            "order": "natural",
            "levels": [8],
            "normalization": False,
            "stats":  ['mean', 'std', 'skew', 'kurtosis'],
        }

EXTRACTION_ARGS = {
            "signals": None,
            "method": "entropy",
            "combined": False,
            "wavelet": "db4",
            "level": 8,
            "order": "natural",
            "levels": [8],
            "normalization": False,
            "stats":  None,
        }

WPD_BEST_IDX = 0
CNN_BEST_IDX = 2

WPD_NAME = ('output', 'LSTM', 'independent', 'FC x1'),
CNN_NAMES = [
        ('output', 'LSTM', False),
        ('output', 'GRU', False),
        ('output', 'LSTM', True),
    ]

OUT_OPTIONS = ['h+o', 'output', 'f_output']
TEMPORAL_TYPE = ['LSTM', 'GRU', 'RNN']
SINGLE_FC_OPTIONS = [True, False]

# Independent = Siamese temporal net
# Independent = single temporal network for concatenated feature vector
FUSION_METHODS = ['independent', 'concatenate']