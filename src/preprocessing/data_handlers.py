import json
import pathlib as pl
import _csv
from abc import ABC, abstractmethod
from typing import Any
import statistics as stat

from utils.types import news_info, words_info, words_dict # type: ignore
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
    def extract(self, row: list[str]):
        """Extract relevant data from source row."""
        pass

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


class WordsDicts(DataHandler):
    """Two dictionaries with included and excluded words, respectively."""
    def __init__(self, to_path: pl.Path, incl_name: str, excl_name: str) -> None:
        """Create empty dicts, store file paths and define destination paths."""
        super().__init__()
        
        self._incl: words_dict = {}
        self._excl: words_dict = {}
        self._incl_stem: words_dict = {}
        self._excl_stem: words_dict = {}

        self._incl_path = to_path / f"{incl_name}.json"
        self._excl_path = to_path / f"{excl_name}.json"
    
    @property
    def all_dicts(self) -> list[words_dict]:
        return [self._incl, self._excl]
    
    @property
    def all_paths(self) -> list[pl.Path]:
        return [self._incl_path, self._excl_path]
    
    @property
    def all_pairs(self) -> list[tuple[pl.Path, words_dict]]:
        return [(path, dct) for path, dct in zip(self.all_paths, self.all_dicts)]

    def extract(self, row: list[str]) -> news_info:
        """Extract type and content from row"""
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
                out_dict = self._excl
                self._n_excl += 1
            else:
                out_dict = self._incl
                self._n_incl += 1
            # Add to relevant dictionary
            for word in words:
                # Add word if it is new
                out_dict[word] = out_dict.get(word, {})
                # Add type if it is new
                out_dict[word][type_] = out_dict[word].get(type_, (0, 0))
                out_dict[word][type_] = add_tuples(
                    out_dict[word][type_],
                    (1 if not word in counted_in_article else 0, 1)
                )
                counted_in_article.add(word)

    def stem_dicts(self) -> None:
        """Stem dicts and combine each into new dict."""
         # Loop through both dicts
        for dct in self.all_dicts:
            old_dct = dct.copy()
            dct.clear()
            # Loop through words
            for tkn in old_dct.keys():
                stemmed_tkn = stem(tkn)
                dct[stemmed_tkn] = dct.get(stemmed_tkn, old_dct[tkn])
                # Loop through frequencies for word
                for type_, freqs in old_dct[tkn].items():
                    dct[stemmed_tkn][type_] = dct[stemmed_tkn].get(type_, (0, 0))
                    current_pair = dct[stemmed_tkn][type_]
                    current_pair = add_tuples(current_pair, freqs)

    def export_json(self) -> None:
        """Dump both dicts as json files."""
        [self.dump_json(*pair) for pair in self.all_pairs]
    
    @classmethod
    def dump_json(cls, file_path: pl.Path, out_dict: dict) -> None:
        """Dump dictionary to json."""
        json_words = json.dumps(out_dict, indent=4)
        with open(file_path, "w") as outfile:
            outfile.write(json_words)

    def finalize(self):
        """Stem, export as JSON and return counts for included and excluded words."""
        self.stem_dicts()
        self.export_json()


class CsvWriter(DataHandler):
    """Class which manages preprocessing and exporting of data on article level."""
    def __init__(self, writer: '_csv._writer') -> None:
        super().__init__()
        self.writer = writer

    def extract(self, row: list[str]) -> tuple[str, ...]:
        """Extract all relevant entries from row."""
        type_ = row[3]
        if type_ is None or type_ in excl_types:
            self._n_excl += 1
            return ()
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

    def write(self, row: list[list[str]]) -> None:
        """Write rows."""
        self.writer.writerows(row)

    def finalize(self):
        """Do nothing."""
        pass