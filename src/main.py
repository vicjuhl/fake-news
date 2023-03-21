import pathlib as pl
import time
import argparse as ap
import numpy as np

from imports.data_importer import (# type: ignore
    extract_words, reduce_corpus, summarize_articles, split_data, remove_stop_words_json, import_val_set, get_split  # type: ignore
)
from imports.json_to_pandas import json_to_pd # type: ignore

def init_argparse() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument("-n", "--nrows", type=int, default=1000)
    parser.add_argument("-f", "--filename", type=str, default="reduced_corpus.csv")
    parser.add_argument("-p", "--processes", nargs="*", type=str)
    parser.add_argument("-q", "--quantiles", nargs=2, type=float)
    parser.add_argument("-v", "--validation_set_num", type=int)
    return parser

if __name__ == "__main__":
    """Run entire pipeline from import to preprocessing to analysis."""

    t0_total = time.time()
    t0 = time.time()

    parser = init_argparse()
    args = parser.parse_args()
    data_path = pl.Path(__file__).parent.parent.resolve() / "data_files/"
    val_set = args.validation_set_num

    if val_set is None or val_set not in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        raise ValueError(f"Validation set value {val_set} is not allowed.")

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

    if not set(args.processes).isdisjoint({"json", "csv", "df"}): # REMOVE df FROM SET TODO (SEE BELOW COMMENT)
        # Load splits information if needed
        splits = get_split(data_path)

    if "json" in args.processes:
        extract_words(
            data_path / "corpus" / args.filename,
            data_path / "words/",
            args.nrows,
            val_set,
            splits,
        )
        print("runtime:", time.time() - t0)
        t0 = time.time()
        
    if "stem_json" in args.processes:
        remove_stop_words_json(
            data_path / f'words/included_words_valset{val_set}.json',
            data_path / f'words/stop_words_removed_valset{val_set}.json',
            *args.quantiles,
        )
        print("runtime:", time.time() - t0)
        t0 = time.time()
        
    if "csv" in args.processes:
        summarize_articles(
            data_path / "corpus" / args.filename,
            data_path / "words" / f"stop_words_removed_valset{val_set}.json",
            data_path / "processed_csv/",
            args.nrows,
            val_set,
            splits,
        )
        print("runtime:", time.time() - t0)
        t0 = time.time()

    if "df" in args.processes:
        # THIS PROCESS SHOULD BE MOVED TO MODEL MAIN SCRIPT WHEN READY TODO
        df = import_val_set(
            data_path / "corpus" / args.filename,
            val_set,
            splits
        )
        print("runtime:", time.time() - t0)
        t0 = time.time()

    print("\n Total runtime:", time.time() - t0_total)