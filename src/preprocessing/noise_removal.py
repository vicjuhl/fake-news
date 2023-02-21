import pandas as pd
from cleantext import clean

def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    df.content = df.content.apply(lambda x: clean(x,
        lower=True,
        normalize_whitespace=True,
        replace_with_url=True,
        replace_with_email=True,
        replace_with_number=True,
        no_punct=True,
    ))
    return df

def tokenize(df: pd.DataFrame) -> tuple[list[str], set[str]]:
    content_tkns: list[list[str]] = [c.split(" ") for c in df["content"]]
    content_tkns_combined = []
    for lst in content_tkns:
        content_tkns_combined += lst
    return (content_tkns_combined, set(content_tkns_combined))
