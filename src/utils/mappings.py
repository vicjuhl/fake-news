import pandas as pd

# Mapping labels to either fake or real unused TODO
labels = {
    "unreliable": "fake",
    "fake": "fake",
    "clickbait": "fake",
    "conspiracy": "fake",
    "reliable": "real",
    "bias": "fake",
    "hate": "fake",
    "junksci": "fake",
    "political": "fake"
}

# Column headers and indexes for input csv
incl_cols = {
    "id": 1,
    "domain": 2,
    "type": 3,
    "url": 4,
    "content": 5,
    "scraped": 6,
    "title": 9,
    "authors": 10,
    "keywords": 11,
    "tags": 14,
    "summary": 15,
}

incl_inds = [ind for ind in incl_cols.values()]
incl_keys = [key for key in incl_cols.keys()]

# Column headers for output csv
out_cols = [
    "id",
    "domain",
    "type",
    "test"
]

# Label types to disregard
excl_types = {
    "satire",
    "unknown",
    ""
}

# Store columns that transfer unchanged from input to output csv's
transfered_cols = [col_name for col_name in incl_keys]

def add_labels(df: pd.DataFrame) -> pd.DataFrame: # Unused TODO
    """Add custom labels based on 'type' column to dataframe."""
    def lookup_labels(data) -> str:
        return labels[data["type"]]
    df["labels"] = df.apply(lookup_labels, axis=1).astype("category")
    return df