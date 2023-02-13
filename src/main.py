import pathlib as pl
import pandas as pd
from data_importer import get_training_data, rinse_training_data

if __name__ == "__main__":
    # parent -> src/, parent.parent -> fake-news/
    file_path = pl.Path(__file__).parent.parent.resolve() / "data_files/news_sample.csv"
    trn_data = rinse_training_data(get_training_data(file_path))
    print(trn_data["type"])
    labels = trn_data.type.unique()
    for label in labels:
        print(label)
