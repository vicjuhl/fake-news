import pathlib as pl
from data_importer import raw_to_words # type: ignore
import argparse as ap
import time

def init_argparse() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument("-n", "--nrows", type=int, default=1000)
    parser.add_argument("-i", "--inclname", type=str, default="included_words")
    parser.add_argument("-e", "--exclname", type=str, default="excluded_words")
    return parser

if __name__ == "__main__":
    """Run entire pipeline from import to preprocessing to analysis."""
    t0 = time.time()

    parser = init_argparse()
    args = parser.parse_args()
    data_path = pl.Path(__file__).parent.parent.resolve() / "data_files"

    file_path = data_path / "news_cleaned_2018_02_13.csv"
    to_path = data_path / "words/"
    n_incl, n_excl, n_skipped = raw_to_words(
        file_path, to_path,
        args.nrows,
        args.inclname,
        args.exclname
    )
    
    print(f"{n_incl + n_excl} rows read, \n {n_incl} were included \n {n_excl} were excluded \n {n_skipped} were skipped \n JSON files were written to {to_path}")
    print("runtime:", time.time() - t0)
    