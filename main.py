import pandas as pd
import numpy as np

from data_cleaning import clean_empty
from linear_regression import gradient_descent

df = pd.read_csv("D3.csv", na_values="########")

df = clean_empty(df)


