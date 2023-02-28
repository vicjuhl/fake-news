import pandas as pd
from cleantext import clean


def cut_tail(df : pd.DataFrame, min_occurence: int) -> pd.DataFrame :
        
    total_words = df.sum(["freq"])   
    ratio = total_words / min_occurence     
    lower_bound = max(ratio , 50)
    acc = 0
    while df["freq"][acc] > lower_bound:
        acc += 1
    
    words_removed = len(df["freq"])-acc
    print("words removed: ", words_removed, "with minimum occurence level: ", min_occurence)
    return df.loc[:acc, :]

print(cut_tail(tokens , 100))
    
 

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
