import pandas as pd
from cleantext import clean
from IPython.display import clear_output


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text for various anomalies for "content" in df."""
    df.content = df.content.apply(lambda x: clean(x,
        lower=True,
        normalize_whitespace=True,
        replace_with_url=True,
        replace_with_email=True,
        replace_with_number=True,
        no_punct=True,
    ))
    return df

def clean_str(text: str) -> str:
    """Clean text for various anomalies."""
    return clean(
        text,
        lower=True,
        normalize_whitespace=True,
        replace_with_url=True,
        replace_with_email=True,
        replace_with_number=True,
        no_punct=True,
    ).replace("\n", "")

def tokenize(df: pd.DataFrame) -> list[str]:
    """Generate list of tokens from dataframe."""
    tkns: list[list[str]] = [c.split(" ") for c in df["content"]]
    tkns_combined = [tkn for section in tkns for tkn in section]
    return tkns_combined

def count_sort(tkns: list[str]) -> pd.DataFrame:
    """Creat dataframe with tokens as rows and frequency as values."""
    counts = dict()
    for tkn in tkns:
        counts[tkn] = counts.get(tkn, 0) + 1
    df = pd.DataFrame.from_dict(counts, orient="index", columns=["freq"])
    df.sort_values(by=["freq"], ascending=False, inplace=True)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run the preprocessing pipeline."""
    df = clean_text(df)
    tkns = tokenize(df)
    counts = count_sort(tkns)
    no_head = counts[50:]
    return no_head