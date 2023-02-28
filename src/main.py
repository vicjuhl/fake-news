import pathlib as pl
import pandas as pd
from data_importer import TrainingData as TD
from preprocessing.noise_removal import preprocess
import argparse as ap

def init_argparse() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default="sample")
    parser.add_argument("-n", "--nrows", type=int, default=500)
    return parser

if __name__ == "__main__":
    """Run entire pipeline from import to preprocessing to analysis."""
    parser = init_argparse()
    args = parser.parse_args()
    if args.file == "full":
        file_name = "news_cleaned_2018_02_13.csv"
    elif args.file == "sample":
        file_name = "news_sample.csv"
    else:
        raise ValueError("Wrong file name alias given")

    file_path = pl.Path(__file__).parent.parent.resolve() / "data_files" / file_name
    trn = TD(file_path, n_rows=args.nrows)
    trn.df, tkns = preprocess(trn.df)
    print(trn.df, tkns)
    