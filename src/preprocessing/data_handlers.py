import json
import pathlib as pl
import _csv
from abc import ABC, abstractmethod
from typing import Any
import statistics as stat
import numpy as np

from utils.types import news_info, words_info, words_dict, NotInTrainingException # type: ignore
from utils.functions import add_tuples, stem # type: ignore
from utils.mappings import transfered_cols, excl_types, incl_cols # type: ignore
from preprocessing.noise_removal import clean_str, tokenize_str # type: ignore


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

    def check_split(self, i: int, id_: int) -> None:
        if self._splits[i, 0] != int(id_): # Sanity check on id numbers
            raise ValueError(f"ID's {(self._splits[i, 0], int(id_))} don't match")
        elif self._splits[i, 1] in {1, self._val_set}: # Discern which set
            raise NotInTrainingException
        
    @classmethod
    @abstractmethod
    def process_batch(cls, data): # TYPING TODO
        """Perform preprocessing on extracted data"""
        pass

    @abstractmethod
    def write(self, articles) -> None: # TYPING TODO
        """Write data to relevant object or file."""
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Do final actions if needed."""
        pass

class CorpusReducer(DataHandler):
    def __init__(self, writer: '_csv._writer') -> None:
        super().__init__()
        self.writer = writer
    
    def extract(self, row: list[str], _) -> tuple[str, ...]:
        """Extract all entries from row."""
        type_ = row[3]
        if type_ is None or type_ in excl_types:
            self._n_excl += 1
            return () # Nothing added to buffer
        else:
            self._n_incl += 1
            # Make sure that every field has data
            return tuple([row[i] for i in range(17)])
        
    @classmethod
    def process_batch(cls, data: list[tuple[str, ...]]) -> list[Any]:
        return data
        
    def write(self, row: list[list[str]]) -> None:
        """Write rows."""
        self.writer.writerows(row)

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
        self.check_split(i, row[1])
        return row[3], row[5]
    
    @classmethod
    def process_batch(cls, data: list[news_info]) -> list[words_info]:
        """Clean text and split into list of type/bag of words pairs."""
        return [(t, tokenize_str(clean_str(c))) for t, c in data]
    
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

    def export_json(self) -> None:
        """Dump both dicts as json files."""
        json_words = json.dumps(self._words, indent=4)
        with open(self._to_path, "w") as outfile:
            outfile.write(json_words)

    def finalize(self):
        """Stem, export as JSON and return counts for included and excluded words."""
        self.stem_dict()
        self.export_json()


class CorpusSummarizer(DataHandler):
    """Class which manages preprocessing and exporting of data on article level."""
    def __init__(self, writer: '_csv._writer', val_set: int, splits: np.ndarray) -> None:
        super().__init__()
        self.writer = writer
        self._val_set = val_set
        self._splits = splits

    def extract(self, row: list[str], i: int) -> tuple[str, ...]:
        """Extract all relevant entries from row."""
        self.check_split(i, row[1])
        type_ = row[3]
        if type_ is None or type_ in excl_types:
            self._n_excl += 1
            return () # Nothing added to buffer
        else:
            self._n_incl += 1
            return tuple(row)

    @classmethod
    def process_batch(cls, data: list[tuple[str, ...]]) -> list[Any]:
        """Transfer specified columns without processing, process others."""
        return_lst = []
        for in_row in data:
            out_row = []
            for col_name in transfered_cols:
                # Add values of transfered cols without processing
                col_index = incl_cols[col_name]
                out_row.append(in_row[col_index])
            # Add values of calculated columns
            content = in_row[5]
            # Shortened article
            cutoff = content.find(" ", 600) # returns -1 if no find, else index of ' '
            short_content = content if cutoff == -1 else content[:cutoff]
            out_row.append(short_content)
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
            return_lst.append(out_row)
        return return_lst

    def write(self, row: list[tuple[str, ...]]) -> None:
        """Write rows."""
        self.writer.writerows(row)

    def finalize(self):
        """Do nothing."""
        pass
