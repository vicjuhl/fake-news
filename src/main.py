import pathlib as pl
from imports.data_importer import (
    extract_words, reduce_corpus, summarize_articles, split_data # type: ignore
)
from imports.json_to_pandas import json_to_pd
import argparse as ap
import time

def init_argparse() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument("-n", "--nrows", type=int, default=1000)
    parser.add_argument("-f", "--filename", type=str, default="reduced_corpus.csv")
    parser.add_argument("-p", "--processes", nargs="*", type=str)
    return parser

if __name__ == "__main__":
    """Run entire pipeline from import to preprocessing to analysis."""
    t0 = time.time()

    parser = init_argparse()
    args = parser.parse_args()
    data_path = pl.Path(__file__).parent.parent.resolve() / "data_files/"

    if "reduce" in args.processes:
        reduce_corpus(
            data_path / "corpus/news_cleaned_2018_02_13.csv",
            data_path / "corpus/",
            args.nrows
        )
        print("runtime:", time.time() - t0)
        t0 = time.time()

    if "split" in args.processes:
        split_data(
            data_path / "corpus" / args.filename,
            data_path / "corpus/"
        )
        print("runtime:", time.time() - t0)
        t0 = time.time()

    if "json" in args.processes:
        extract_words(
            data_path / "corpus" / args.filename,
            data_path / "words/",
            args.nrows,
        )
        print("runtime:", time.time() - t0)
        t0 = time.time()
        
    if "csv" in args.processes:
        summarize_articles(
            data_path / "corpus" / args.filename,
            data_path / "processed_csv/",
            args.nrows
        )
        print("runtime:", time.time() - t0)
        t0 = time.time()

    if "df" in args.processes:
        df = json_to_pd()
        print("runtime:", time.time() - t0)
        t0 = time.time()