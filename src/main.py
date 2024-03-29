import pathlib as pl
import time
import argparse as ap
import numpy as np

from imports.data_importer import ( # type: ignore
    extract_words,
    reduce_corpus,
    summarize_articles,
    split_data,
    remove_stop_words_json,
    shorten_articles,
    get_split,
    get_duplicate_ids
)

procs = {"reduce", "shorten", "split", "json", "stem_json", "summarize", "get_dups"}

def init_argparse() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument("-n", "--nrows", type=int, default=1000, help="Number of rows to include")
    parser.add_argument("-f", "--filename", type=str, default="reduced_corpus.csv", help="Path to reduced corpus")
    parser.add_argument("-p", "--processes", nargs="*", type=str, choices=procs, help="Processes to run; see below if cases")
    parser.add_argument("-q", "--quantiles", nargs=2, type=float, help="How much to cut head and tail: give two floats; each one fraction of total word count to reduce, for instance 0.5 0.05 will cut 50 percent off the bottom and 5 percent off the top.")
    parser.add_argument("-v", "--validation_set_num", type=int, help="Validation set split number chosen for this pipeline.")
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
            data_path / "corpus/duplicates.csv",
            data_path / "corpus/",
            args.nrows
        )
        print("runtime:", time.time() - t0)
        t0 = time.time()
    
    if "shorten" in args.processes:
        shorten_articles(
            data_path / "corpus/reduced_corpus.csv",
            data_path / "processed_csv/",
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

    if not set(args.processes).isdisjoint({"json", "summarize"}):
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
            val_set,
            data_path / f'words/stop_words_removed_valset{val_set}.json',
            *args.quantiles,
        )
        print("runtime:", time.time() - t0)
        t0 = time.time()

    if "summarize" in args.processes:
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

    if "get_dups" in args.processes:
        get_duplicate_ids(
            data_path / f"processed_csv/shortened_corpus.csv",
            data_path / "corpus",
            f"duplicates.csv" # this name has been hard-coded
        )
        print("runtime:", time.time() - t0)
        t0 = time.time()

    print("\n Total runtime:", time.time() - t0_total)