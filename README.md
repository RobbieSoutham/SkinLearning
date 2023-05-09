# MastersDiss
Deep learning + feature extraction approaches to inferring the material characteristics of skin.
Dataset obtained with FEBio using Quasi-monte Carlo with Sobol sequences.

# Best approaches
## CNN
- Upsampling
- LSTM + entire output state
- Sequence dimension is number of filters

## WPD
- Statistical significance over CNN
- Sequence length of 1 -> LSTM not fully utilised
- Downstreamed to standard FFNN (ReLU) or a CNN performs worse
- Should test FFNN with tahn activation

# Skin model
- Potentially try just the loading phase
- Temporal dependencies may come into play if the deloading phase presents better behavior or is removed

# Further improvments
- May benefit from optimising WPD parameters (BO)
- May benefit from reduction in learning rate (increased computational cost)
- Save the model during early stopping to ensure the best model seen during training is kept


## Directory structure
- `SkinLearning/`
    -`Datset/`
        - Code to generate databases.
    - `Experiments/`
        - Code to sample architecture combinations.
    - `Utils/`
        - Generic dataset and other utility functions.
    - `NN/`
        - `Models.py`
            - Classes for the types of models.
            - Best model locations.
        - `Helpers.py`
            - Train and test functions.
            - Default patients based on trail and error.
            - MAE is used as better performance was experienced.
