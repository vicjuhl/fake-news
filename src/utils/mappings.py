import pandas as pd

# Column headers and indexes for input csv
incl_cols = {
    "id": 1,
    "domain": 2,
    "orig_type": 3,
    "url": 4,
    "content": 5,
    "scraped_at": 6,
    "title": 9,
    "authors": 10,
    "keywords": 11,
    "tags": 14,
    "summary": 15,
    "type": 16,
}

incl_inds = [ind for ind in incl_cols.values()]
incl_keys = [key for key in incl_cols.keys()]

# Column headers for output csv
out_cols = [
    # Transfered
    "id",
    "domain",
    "type",
    "scraped_at",
    # Derived
    "words",
    "content_len",
    "mean_word_len",
    "median_word_len",
    "split",
]
# Store columns that transfer unchanged from input to output csv's
transfered_cols: list[str] = []
for col_name in incl_keys:
    if col_name in out_cols:
        transfered_cols.append(col_name)

# Label types to disregard
excl_types = {
    "satire",
    "conspiracy",
    "clickbait",
    "unreliable",
    "political",
    "state",
    "unknown",
    "rumor",
    ""
}

# Our chosen labels and their respective groupings
labels = {
    "fake": "fake",
    "bias": "fake",
    "junksci": "fake",
    "hate": "fake",
    "reliable": "reliable"
}

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add column to dataframe with label groups based on 'type' column."""
    df = df.rename(columns={"type":"orig_type"})
    
    def lookup_labels(data) -> str:
        return labels[data["orig_type"]]
    df["type"] = df.apply(lookup_labels, axis=1)

    new_index = ["id", "domain", "orig_type","type", "scraped_at", "words", "content_len", "mean_word_len", "median_word_len", "split"]
    df = df.reindex(columns= new_index)
    return df