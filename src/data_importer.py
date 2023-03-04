import pathlib as pl
import pandas as pd
from preprocessing.noise_removal import clean_str # type: ignore
import csv
import json

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

def dump_json(file_path: pl.Path, out_dict: dict) -> None:
    """Dump dictionary to json."""
    json_words = json.dumps(out_dict, indent=4)
    with open(file_path, "w") as outfile:
        outfile.write(json_words)

def raw_to_words(from_file: pl.Path, to_path: pl.Path, n_rows: int) -> int:
    """Read raw csv file line by line, clean words, count occurrences and dump to json."""
    words: dict[str, dict[str, int]] = {} # Included words
    excl: dict[str, dict[str, int]] = {} # Excludes words
    skipped: int = 0 # Count skipped rows that could not be read.
    to_path.mkdir(parents=True, exist_ok=True) # Create dest folder if it does not exist
    
    with open(from_file) as f:
        reader = csv.reader(f)
        next(reader) # skip header

        for i, row in enumerate(reader):
            if i % 5000 == 0:
                print("Processed lines: ", i, "...") # "progress bar"
            try:
                # Skip row if either type or content is not well defined
                type_ = row[3]
                content = row[5]
            except:
                skipped += 1
                continue

            to_dict = excl if type_ is None or type_ in ["satire", "unknown"] else words
            content_clean = clean_str(content)
            tkns = content_clean.split(" ")

            # Increment total counter and type counter for word
            for tkn in tkns:
                to_dict[tkn] = to_dict.get(tkn, {})
                to_dict[tkn][type_] = to_dict[tkn].get(type_, 0) + 1
            
            # Break when target rows reached
            if i == n_rows:
                break

    # Export as json
    dump_json(to_path / "words.json", words)
    dump_json(to_path / "excluded_words.json", excl)

    return skipped