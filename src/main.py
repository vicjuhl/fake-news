import pathlib as pl
import pandas as pd
from data_importer import TrainingData as TD

if __name__ == "__main__":
    # parent -> src/, parent.parent -> fake-news/
    file_path = pl.Path(__file__).parent.parent.resolve() / "data_files/news_sample.csv"
    trn = TD(file_path)
    print(trn.df["type"])
    labels = trn.df.type.unique()
    for label in labels:
        print(label)
