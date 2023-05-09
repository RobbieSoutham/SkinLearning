# MastersDiss
Deep learning + feature extraction approaches to inferring the material characteristics of skin.
Dataset obtained with FEBio simulations of the Cutometer using Quasi-monte Carlo with Sobol sequences.
Best parameters stored in SkinLEarning/Experiments/ModelArgs

## Best approaches
Best parameters stored in SkinLEarning/Experiments/ModelArgs

#### CNN Feature Extraction
- Upsampling - capturing higher level features significantly more effective
- Larger kernels to begin with
- Dual channel - process curves individually

#### WPD Feature Extraction
- Only use the final level from decomposition
- Decompose signals independently
- Concatenate sub-bands
- No significant improvement past maximum decomposition level of 8
-  Use mean, standard deviation, skew, kurtosis
    - Energy, entropy, or other combinations of statistics no better than raw coefficients
- Daubechies 4 (Db 4)
    - Symlets 3, 4, 6 and db 2, 6 showed no improvements

### CNN-LSTM
- LSTM + entire output state
- Sequence dimension is number of filters
- Proccessed together

#### WPD-LSTM
- Statistical significance over CNN-LSTM
- Siamese LSTM
- Sequence length of 1 -> LSTM not fully utilised
- Downstreamed to standard FFNN (ReLU) or a CNN performs worse
- Should test FFNN with tahn activation
- Hidden size = 2 * total number of features

## Skin model
- Potentially try just the loading phase
- Temporal dependencies may come into play if the deloading phase presents better behavior or is removed

## Further improvments
- May benefit from optimising WPD parameters (BO)
- May benefit from reduction in learning rate (increased computational cost)
- Save the model during early stopping to ensure the best model seen during training is kept

## Dataset
- https://www.dropbox.com/s/x98fzkr7ebqthfr/SamplingResults.zip?dl=0
- Utils/SetupDataset.py for creating and saving WPD datasets

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
