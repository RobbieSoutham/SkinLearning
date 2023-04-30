TOP_WPD_ARGS = [
        {
            'conv': False,
            'out': 'h+o',
            'temporal_type': 'LSTM',
            'single_fc': False,
            'input_size': 512,
            'hidden_size': 1024,
            'fusion_method': 'independent'
        },
        {
            'conv': False,
            'out': 'f_output',
            'temporal_type': 'GRU',
            'single_fc': False,
            'input_size': 512,
            'hidden_size': 1024,
            'fusion_method': 'independent'
        },
        {
            'conv': False,
            'out': 'f_output',
            'temporal_type': 'LSTM',
            'single_fc': False,
            'input_size': 512,
            'hidden_size': 1024,
            'fusion_method': 'independent'
        }
]

TOP_CNN_ARGS = [
        {
            'out': 'output',
            'temporal_type': 'LSTM',
            'single_fc': False,
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

EXTRACTION_ARGS = {
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

WPD_NAMES = [
        ('h+o', 'LSTM', 'independent', 'FC x6'),
        ('f_output', 'GRU', 'independent', 'FC x6'),
        ('f_output', 'LSTM', 'independent', 'FC x6'),
    ]
CNN_NAMES = [
        ('output', 'LSTM', False),
        ('output', 'GRU', False),
        ('output', 'LSTM', True),
    ]