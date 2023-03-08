import pandas as pd

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

incl_cols = {
    "id": 1,
    "domain": 2,
    "type_": 3,
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

excl_types = {
    "satire",
    "unknown",
    ""
}

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add custom labels based on 'type' column to dataframe."""
    def lookup_labels(data) -> str:
        return labels[data["type"]]
    df["labels"] = df.apply(lookup_labels, axis=1).astype("category")
    return df