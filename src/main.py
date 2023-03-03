import pathlib as pl
from data_importer import TrainingData, raw_to_words # type: ignore
from preprocessing.noise_removal import preprocess # type: ignore
import argparse as ap

def init_argparse() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default="sample")
    parser.add_argument("-n", "--nrows", type=int, default=1000)
    return parser

if __name__ == "__main__":
    """Run entire pipeline from import to preprocessing to analysis."""
    parser = init_argparse()
    args = parser.parse_args()
    if args.file == "full":
        file_path = pl.Path(__file__).parent.parent.resolve() / "data_files/news_cleaned_2018_02_13.csv"
        to_path = pl.Path(__file__).parent.parent.resolve() / "data_files/words/"
        skipped = raw_to_words(file_path, to_path, args.nrows)
        print(f"Data written as json to {to_path}, skipped {skipped} rows")
    
    elif args.file == "sample":
        file_path = pl.Path(__file__).parent.parent.resolve() / "data_files/news_sample.csv"
        trn = TrainingData(file_path, n_rows=args.nrows)
        trn.df = preprocess(trn.df)
        print(trn.df)
    
    else:
        raise ValueError("Wrong file name alias given")
    