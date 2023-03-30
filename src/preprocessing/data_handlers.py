import json
import pathlib as pl
import _csv
from abc import ABC, abstractmethod
from typing import Any
import statistics as stat
import numpy as np
import pandas as pd

from utils.types import news_info, words_info, words_dict, NotInTrainingException # type: ignore
from utils.functions import add_tuples # type: ignore
from utils.mappings import transfered_cols, excl_types, incl_cols, labels # type: ignore
from preprocessing.noise_removal import (  # type: ignore
    clean_str,
    tokenize_str,
    stem,
    preprocess_string,
    preprocess_without_stopwords
)

class DataHandler(ABC):
    """Abstract class for data object such as dictionaries of words or csv-writers."""
    def __init__(self) -> None:
        self._n_incl: int = 0
        self._n_excl: int = 0

    @property
    def n_incl(self) -> int:
        return self._n_incl

    @property
    def n_excl(self) -> int:
        return self._n_excl
    
    @abstractmethod
    def extract(self, row: list[str], i: int):
        """Extract relevant data from source row."""
        pass

    def check_split(self, i: int, id_: int, splits: np.ndarray, val_set: int) -> None:
        """Raise errors if id's don't match on lookup or row not in training set."""
        if splits[i, 0] != id_: # Sanity check on id numbers
            raise ValueError(f"ID's {(splits[i, 0], int(id_))} don't match")
        elif splits[i, 1] in {1, val_set}: # If in val or test set
            raise NotInTrainingException
        
    @classmethod
    @abstractmethod
    def process_batch(cls, data):
        """Perform preprocessing on extracted data"""
        pass

    @abstractmethod
    def write(self, articles) -> None:
        """Write data to relevant object or file."""
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Do final actions if needed."""
        pass


class CorpusReducer(DataHandler):
    def __init__(self, writer: '_csv._writer', duplicates: np.ndarray) -> None:
        super().__init__()
        self.writer = writer
        self.duplicates = duplicates
    
    def extract(self, row: list[str], _) -> tuple[str, ...]:
        """Extract all entries from row except duplicates."""
        type_ = row[3]
        id_ = int(row[1])
        if type_ is None or type_ in excl_types:
            self._n_excl += 1
            return () # Nothing added to buffer
        elif id_ in self.duplicates:
            self._n_excl += 1
            return () # Nothing added to buffer
        else:
            self._n_incl += 1
            # Make sure that every field has data
            return tuple([row[i] for i in range(len(row))])
        
    @classmethod
    def process_batch(cls, data: tuple[list[tuple[str, ...]], dict]) -> list[Any]:
        batch, _ =  data
        result_lst = []
        for row in batch:
            out_row = list(row)
            out_row.append(labels[row[3]])
            result_lst.append(out_row)
        return result_lst
        
    def write(self, row: list[list[str]]) -> None:
        """Write rows."""
        self.writer.writerows(row)

    def finalize(self):
        """Do nothing."""
        pass


class CorpusShortener(DataHandler):
    def __init__(self, writer: '_csv._writer') -> None:
        super().__init__()
        self._writer = writer
    
    def extract(self, row: list[str], _) -> tuple[str, ...]:
        id_ = row[1]
        content = row[5]
        self._n_incl += 1
        return (id_, content)
    
    @classmethod
    def process_batch(cls, data: tuple[list[tuple[str, ...]], dict]) -> list[Any]:
        """Get id and shorten content."""
        batch, _ = data
        ret = [[row[0], row[1][:500]] for row in batch]
        return ret

    def write(self, rows: list[tuple[str, ...]]) -> None:
        """Write rows."""
        self._writer.writerows(rows)

    def finalize(self):
        """Do nothing."""
        pass


class WordsCollector(DataHandler):
    """Two dictionaries with included and excluded words, respectively."""
    def __init__(self, to_path: pl.Path, val_set: int, splits: np.ndarray) -> None:
        """Create empty dicts, store file paths and define destination paths."""
        super().__init__()
        self._words: words_dict = {}
        self._to_path = to_path
        self._val_set = val_set
        self._splits = splits

    def extract(self, row: list[str], i: int) -> news_info:
        """Extract type and content from row"""
        self.check_split(i, int(row[1]), self._splits, self._val_set)
        return row[incl_cols["type"]], row[5]
    
    @classmethod
    def process_batch(cls, data: tuple[list[news_info], dict]) -> list[words_info]:
        """Clean text and split into list of type/bag of words pairs."""
        batch, _ = data
        return [(t, tokenize_str(clean_str(c))) for t, c in batch]
    
    def write(self, articles: list[words_info]):
        """Add article as bag of words counts to relevant dictionary."""
        for type_, words in articles:
            # Keep track of words already counted in current article
            counted_in_article: set[str] = set()
            # Decide where to add word based on type
            if type_ is None or type_ in excl_types:
                self._n_excl += 1
            else:
                self._n_incl += 1
            # Add to relevant dictionary
            for word in words:
                # Add word if it is new
                self._words[word] = self._words.get(word, {})
                # Add type if it is new
                self._words[word][type_] = self._words[word].get(type_, (0, 0))
                self._words[word][type_] = add_tuples(
                    self._words[word][type_],
                    (1 if not word in counted_in_article else 0, 1)
                )
                counted_in_article.add(word)

    def stem_dict(self) -> None:
        """Stem dicts and combine each into new dict."""
         # Loop through both dicts
        print("Number of words before stemming:", len(self._words.keys()))
        old_dct = self._words.copy()
        self._words.clear()
        # Loop through words
        for tkn in old_dct.keys():
            stemmed_tkn = stem(tkn)
            self._words[stemmed_tkn] = self._words.get(stemmed_tkn, old_dct[tkn])
            # Loop through frequencies for word
            for type_, freqs in old_dct[tkn].items():
                self._words[stemmed_tkn][type_] = self._words[stemmed_tkn].get(type_, (0, 0))
                current_pair = self._words[stemmed_tkn][type_]
                current_pair = add_tuples(current_pair, freqs)
        print("Number of words after stemming:", len(self._words.keys()))

    def export_json(self, data) -> None:
        """Dump both dicts as json files."""
        json_words = json.dumps(data, indent=4)
        with open(self._to_path, "w") as outfile:
            outfile.write(json_words)

    def finalize(self):
        """Stem, export as JSON and return counts for included and excluded words."""
        self.stem_dict()
        data = {"nArticles": self.n_incl, "words": self._words} #extract article count
        self.export_json(data)


class CorpusSummarizer(DataHandler):
    """Class which manages preprocessing and exporting of data on article level."""
    def __init__(
        self,
        summ_writer: '_csv._writer',
        val_set: int,
        splits: np.ndarray,
        file_path: pl.Path
    ) -> None:
        super().__init__()
        self.summ_writer = summ_writer
        self._val_set = val_set
        self._splits = splits
        self._file_path = file_path

    def extract(self, row: list[str], i: int) -> tuple[str, ...]:
        """Extract all relevant entries from row."""
        self.check_split(i, int(row[1]), self._splits, self._val_set)
        type_ = row[incl_cols["type"]]
        if type_ is None or type_ in excl_types:
            self._n_excl += 1
            return () # Nothing added to buffer
        else:
            self._n_incl += 1
            split_num = self._splits[i, 1]
            # Add split number as the last entry for later writing
            row.append(str(split_num))
            return tuple(row)

    @classmethod
    def process_batch(cls, data: tuple[list[tuple[str, ...]], dict]) -> list[list[Any]]:
        """Transfer specified columns without processing, process others."""
        # Unpack and prepare
        batch, kwargs = data
        incl_words = kwargs["incl_words"]
        return_lst: list[list[Any]] = []
        # Iterate through batch
        for in_row in batch:
            out_row = []
            for col_name in transfered_cols:
                # Add values of transfered cols without processing
                col_index = incl_cols[col_name]
                out_row.append(in_row[col_index])
            # Add values of calculated columns
            content = in_row[5]
            # Bag of words
            out_row.append(preprocess_without_stopwords(content, incl_words))
            # Length of content
            out_row.append(len(content))
            # Mean token length
            tkns = tokenize_str(clean_str(content))
            tkns_lens = [len(tkn) for tkn in tkns]
            mean_len = sum(tkns_lens)/float(len(tkns))
            out_row.append(mean_len)
            # Median token length
            median_len = stat.median(tkns_lens)
            out_row.append(median_len)
            # Split number
            out_row.append(in_row[-1])
            # Append to return list
            return_lst.append(out_row)
        return return_lst

    def write(self, rows: list[tuple[str, ...]]) -> None:
        """Write rows."""
        self.summ_writer.writerows(rows)

    def finalize(self):
        """Shuffle the deck."""
        print("Shuffling summarized corpus.")
        df = pd.read_csv(self._file_path)
        df = df.sample(frac=1.0, random_state=42)
        df.to_csv(self._file_path, index=False)

class ValidationProcessor(DataHandler):
    """Preprocess the validation data.
    
    Assuming all rows are good and filtered for duplicates.
    """
    def __init__(
        self,
        df_val_data: pd.DataFrame,
        splits: pd.DataFrame,
        to_file: pl.Path,
    ) -> None:
        super().__init__()
        self._df_val_data = df_val_data
        self._splits = splits
        self._to_file = to_file

    def extract(self, row: list[str], i: int):
        """Extract relevant fields from row."""
        id_ = int(row[incl_cols["id"]])
        in_split = id_ in self._splits
        if in_split:
            orig_type = row[incl_cols["orig_type"]]
            type_ = row[incl_cols["type"]]
            content = row[incl_cols["content"]]
            self._n_incl += 1
            return (id_, orig_type, type_, content)
        else:
            self._n_excl += 1
            return ()
    
    @classmethod
    def process_batch(cls, data: tuple[list[tuple[str, ...]], dict]) -> list[list[Any]]:
        batch, _ = data

        return_lst: list[list] = []
        for row in batch:
            id_ = int(row[0])
            orig_type = row[1]
            type_ = row[2]
            content = row[3]
            out_row: list[Any] = [id_]
            out_row.append(orig_type)
            out_row.append(type_)
            out_row.append(preprocess_string(content))
            return_lst.append(out_row)
        return return_lst
    
    def write(self, rows: list[list]) -> None:
        buffer_df = pd.DataFrame(rows, columns=["id", "orig_type", "type", "words"])
        self._df_val_data = pd.concat([self._df_val_data, buffer_df])

    def finalize(self) -> None:
        self._df_val_data = self._df_val_data.sample(frac=1.0, random_state=42)
        self._df_val_data.to_csv(self._to_file, index=False)