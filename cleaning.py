import pandas as pd
import numpy as np

def clean_empty(df: pd.DataFrame) -> pd.DataFrame:
    # Replaces empty cells with the average value of the column
    df = df.apply(pd.to_numeric, errors='coerce')  
    df.fillna(df.mean(), inplace=True)
    return df
    
