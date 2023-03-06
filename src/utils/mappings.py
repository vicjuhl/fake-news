import pandas as pd

excluded = {
    "satire",
    "unknown",
    ""
}

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

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add custom labels based on 'type' column to dataframe."""
    def lookup_labels(data) -> str:
        return labels[data["type"]]
    df["labels"] = df.apply(lookup_labels, axis=1).astype("category")
    return df