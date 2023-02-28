import pathlib as pl
import pandas as pd

dtypes = {
    "id": int,
    "domain": str,
    "type": str,
    "url": str,
    "content": str,
    "scraped_at": str,
    "inserted_at": str,
    "updated_at": str,
    "title": str,
    "authors": str,
    "keywords": str,
    "meta_keywords": str,
    "meta_description": str,
    "tags": str,
    "summary": str 
}

class TrainingData:
    def __init__(self, file_path: pl.Path, n_rows: int = 100) -> None:
        """Instantiate TrainingData class object."""
        self.df = self.get_training_data(file_path, n_rows)
        self.rinse_training_data()
        # self.add_labels()

    def get_training_data(self, file_path: pl.Path, n_rows: int) -> pd.DataFrame:
        """Import and typecast training data from csv file."""
        df = pd.read_csv(file_path, nrows=n_rows, dtype=dtypes)
        return df

    def rinse_training_data(self) -> None:
        """Rinse training data."""
        df = self.df[self.df['type'].notna()] # Remove unclassified rows
        self.df = df

    def add_labels(self) -> None:
        """Add custom labels based on 'type' column."""
        labels = {
            "unreliable": "fake",
            "fake": "fake",
            "clickbait": "fake",
            "conspiracy": "fake",
            "reliable": "real",
            "bias": "fake",
            "hate": "fake",
            "junksci": "fake",
            "political": "fake",
            "unknown": "unknown"
        }
        def lookup_labels(data) -> str:
            return labels[data["type"]]
        self.df["labels"] = self.df.apply(lookup_labels, axis=1).astype("category")