from data_loader import load_data
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

set_config(transform_output='pandas')

train = load_data('train_merged.parquet')

np.random.seed(0)

cols_fill_with_zeros = ['transactions_lag1']

num_zero_pipe = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant', fill_value=0))
])

