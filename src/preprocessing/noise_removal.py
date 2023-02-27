import numpy as np
import pandas as pd
from cleantext import clean

def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text for various anomalies."""
    df.content = df.content.apply(
        lambda x: clean(
            x,
            lower=True,
            normalize_whitespace=True,
            replace_with_url=True,
            replace_with_email=True,
            replace_with_number=True,
            no_punct=True,
        ).replace("\n", " ")
    )
    return df

def tokenize(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add tokens column to df and create combined list of tokens from df."""
    df["tokens"] = df.content.apply(lambda c: c.split(" "))
    df["n_tokens"] = df.tokens.apply(lambda tkns: len(tkns))
    tkns: list[list[str]] = [c.split(" ") for c in df["content"]]
    tkns_combined = [tkn for section in tkns for tkn in section]
    return df, tkns_combined

def count_sort(tkns: list[str]) -> pd.DataFrame:
    """Create dataframe with tokens as rows and frequency as values."""
    counts: dict[str, int] = dict()
    for tkn in tkns:
        counts[tkn] = counts.get(tkn, 0) + 1
    df = pd.DataFrame.from_dict(counts, orient="index", columns=["freq"])
    df.sort_values(by=["freq"], ascending=False, inplace=True)
    return df

def search_tokens(tkns_srs: pd.Series, text: str) -> None:
    """Find text in Series of tokens"""
    for tokens in tkns_srs:
        for token in tokens:
            if text in token:
                print(token)

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the preprocessing pipeline."""
    df = clean_text(df)
    df, tkns = tokenize(df)
    counts = count_sort(tkns)
    no_head = counts[50:]
    search_tokens(df["tokens"], "ss")
    return df, no_head