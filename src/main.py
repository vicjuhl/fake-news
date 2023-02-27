import pathlib as pl
import pandas as pd
from data_importer import TrainingData as TD
from preprocessing.noise_removal import preprocess

if __name__ == "__main__":
    """Run entire pipeline from import to preprocessing to analysis."""
    file_path = pl.Path(__file__).parent.parent.resolve() / "data_files/news_sample.csv"
    trn = TD(file_path)
    trn.df, tkns = preprocess(trn.df)
    print(trn.df, tkns)
    