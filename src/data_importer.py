import pathlib as pl
import pandas as pd

class TrainingData:
    def __init__(self, file_path: pl.Path) -> None:
        """Instantiate TrainingData class object."""
        self.df = self.get_training_data(file_path)
        self.rinse_training_data()
        self.add_labels()

    def get_training_data(self, file_path: pl.Path) -> pd.DataFrame:
        """Import and typecast training data from csv file."""
        df = pd.read_csv(file_path)
        df = df.drop(df.columns[0], axis=1)
        df["type"] = pd.Categorical(df.type)
        return df

    def rinse_training_data(self) -> None:
        """Rinse training data."""
        df = self.df[self.df['type'].notna()]
        self.df = df

    def add_labels(self) -> None:
        """Add custom labels based on 'type' column."""
        labels = {
            "unreliable": "unknown",
            "fake": "fake",
            "clickbait": "unknown",
            "conspiracy": "fake",
            "reliable": "real",
            "bias": "unknown",
            "hate": "unknown",
            "junksci": "fake",
            "political": "unknown",
            "unknown": "unknown"
        }
        def lookup_labels(data) -> str:
            return labels[data["type"]]
        self.df["labels"] = self.df.apply(lookup_labels, axis=1).astype("category")