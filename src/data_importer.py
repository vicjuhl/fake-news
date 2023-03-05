import pathlib as pl
import pandas as pd
from preprocessing.noise_removal import clean_str # type: ignore
import csv
import json
from multiprocessing import Pool, pool
import sys

news_info  = tuple[str, str] # type and content
words_info = tuple[str, list[str]] # type and words
words_dict = dict[str, dict[str, int]] # words and {type: count}

csv.field_size_limit(sys.maxsize)

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

class WordsDicts:
    """Two dictionaries with included and excluded words, respectively."""
    def __init__(self, to_path: pl.Path, incl_name: str, excl_name: str) -> None:
        """Create empty dicts, store file paths and make destination folder."""
        self._incl: words_dict = {}
        self._excl: words_dict = {}
        self._incl_path = to_path / f"{incl_name}.json"
        self._excl_path = to_path / f"{excl_name}.json"
        to_path.mkdir(parents=True, exist_ok=True) # Create dest folder if it does not exist
        self._n_incl: int = 0
        self._n_excl: int = 0

    @property
    def n_incl(self) -> int:
        return self._n_incl

    @property
    def n_excl(self) -> int:
        return self._n_excl

    def add_words(self, data_list: list[words_info]) -> None:
        """Add bag of words to self."""
        for type_, words in data_list:
            # Decide where to add word based on type
            if type_ is None or type_ in ["satire", "unknown", ""]:
                out_dict = self._excl
                self._n_excl += 1
            else:
                out_dict = self._incl
                self._n_incl += 1
            # Add to relevant dictionary
            for word in words:
                out_dict[word] = out_dict.get(word, {}) # Add word if it is new
                out_dict[word][type_] = out_dict[word].get(type_, 0) + 1 # Add one
        
    def export_json(self) -> None:
        """Dump both dicts as json files."""
        self.dump_json(self._incl_path, self._incl)
        self.dump_json(self._excl_path, self._excl)

    @classmethod
    def dump_json(cls, file_path: pl.Path, out_dict: dict) -> None:
        """Dump dictionary to json."""
        json_words = json.dumps(out_dict, indent=4)
        with open(file_path, "w") as outfile:
            outfile.write(json_words)

def process_batch(articles: list[news_info]) -> list[words_info]:
    return [(t, clean_str(c).split(" ")) for t, c in articles]

def create_clear_buffer(n_procs: int) -> list[list[news_info]]:
    buffer: list[list[news_info]] = []
    for _ in range(n_procs):
        buffer.append([])
    return buffer

def process_buffer(buffer: list[list[news_info]], n_procs: int) -> list[words_info]:
    with Pool(n_procs) as p:
        data_results = p.map_async(process_batch, buffer)
        data_lists = data_results.get()
        # Concattenate list of lists of words to just list of word_info
        return [article for batch in data_lists for article in batch]

def raw_to_words(
    from_file: pl.Path,
    to_path: pl.Path,
    n_rows: int,
    incl_name: str,
    excl_name: str
) -> tuple[int, int, int]:
    """Read raw csv file line by line, clean words, count occurrences and dump to json.
    
    Return tuple of n_read, n_skipped
    """
    out_dicts = WordsDicts(to_path, incl_name, excl_name) # Included words

    n_read: int = 0 # Count rows that were be read.
    n_skipped: int = 0 # Count skipped rows that could not be read.
    n_rows -= 1 # Compensate for 0-indexing

    n_procs = 8 # DYNAMIC TODO
    batch_sz = 1000
    buffer_sz = n_procs * batch_sz
    # n empty lists
    buffer = create_clear_buffer(n_procs)

    running = True
    i = 0
    
    with open(from_file) as f:
        reader = csv.reader(f)
        next(reader) # skip header
        row = next(reader)

        while running:
            try:
                row = next(reader)
                if i % 5000 == 0:
                    print("Processed lines:", i, "...") # "progress bar"

                # Parallel process data if all batches are full
                if n_read % buffer_sz == 0:
                    out_dicts.add_words(process_buffer(buffer, n_procs))
                    buffer = create_clear_buffer(n_procs)

                # Read and save line
                try:
                    type_ = row[3]
                    content = row[5]
                    # Add article to appropriate batch
                    buffer_index = (n_read % buffer_sz)//batch_sz
                    batch = buffer[buffer_index]
                    batch.append((type_, content))
                    n_read += 1
                # Or skip row if either type or content cannot be read
                except:
                    n_skipped += 1 # ERROR HERE TODO
                    print("batch add fail", i, n_skipped)
                
                # Break when target rows reached
                if i >= n_rows:
                    running = False
                i += 1

            except: # No row to read (or other error)
                running = False
        
        # Flush what remains in the buffer
        out_dicts.add_words(process_buffer(buffer, n_procs))

    # Export as json
    out_dicts.export_json()

    return out_dicts.n_incl, out_dicts.n_excl, n_skipped