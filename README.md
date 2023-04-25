# MastersDiss
Development of a machine learning model able to predict the material characteristics of skin using suction experiments.

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
