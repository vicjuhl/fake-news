import pathlib as pl
import pandas as pd

def get_training_data(file_path: pl.Path) -> pd.DataFrame:
    """Import training data from csv file."""
    df = pd.read_csv(file_path)
    df = df.drop(df.columns[0], axis=1)
    df["type"] = pd.Categorical(df.type)
    return df

def rinse_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Rinse training data."""
    df = df[df['type'].notna()]
    return df

