import pathlib as pl
from data_importer import raw_to_words # type: ignore
import argparse as ap
import time

def init_argparse() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument("-n", "--nrows", type=int, default=1000)
    parser.add_argument("-i", "--inclname", type=str, default="included_words")
    parser.add_argument("-e", "--exclname", type=str, default="excluded_words")
    parser.add_argument("-p", "--processes", nargs="*", type=str)
    return parser

if __name__ == "__main__":
    """Run entire pipeline from import to preprocessing to analysis."""
    t0 = time.time()

    parser = init_argparse()
    args = parser.parse_args()
    data_path = pl.Path(__file__).parent.parent.resolve() / "data_files"

    from_file = data_path / "news_cleaned_2018_02_13.csv"
    to_path = data_path

    if "json" in args.processes:
        raw_to_words(
            from_file,
            to_path / "words/",
            args.nrows,
            args.inclname,
            args.exclname
        )
        print("runtime:", time.time() - t0)
        t0 = time.time()
        
    if "csv" in args.processes:
        reduce_raw(from_file, to_path / "processed_csv/", args.nrows)
        print("runtime:", time.time() - t0)
        t0 = time.time()

    