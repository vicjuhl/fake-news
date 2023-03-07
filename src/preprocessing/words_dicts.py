import json

import pathlib as pl
from utils.types import words_info, words_dict # type: ignore
from nltk import PorterStemmer # type: ignore

ps = PorterStemmer()

class WordsDicts:
    """Two dictionaries with included and excluded words, respectively."""
    def __init__(self, to_path: pl.Path, incl_name: str, excl_name: str) -> None:
        """Create empty dicts, store file paths and make destination folder."""
        self._incl: words_dict = {}
        self._excl: words_dict = {}
        self._incl_stem: words_dict = {}
        self._excl_stem: words_dict = {}

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
    
    @property
    def all_dicts(self) -> list[words_dict]:
        return [self._incl, self._excl]
    
    @property
    def all_paths(self) -> list[pl.Path]:
        return [self._incl_path, self._excl_path]
    
    @property
    def all_pairs(self) -> list[tuple[pl.Path, words_dict]]:
        return [(path, dct) for path, dct in zip(self.all_paths, self.all_dicts)]

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
        [self.dump_json(*pair) for pair in self.all_pairs]

    def stem(self) -> None:
        """Stem dicts and combine each into new dict."""
         # Loop through both dicts
        for dct in self.all_dicts:
            old_dct = dct.copy()
            dct.clear()
            # Loop through words
            for tkn in old_dct.keys():
                stemmed_tkn = ps.stem(tkn)
                try:
                    # Loop through frequencies for word
                    for type_, freq in old_dct[tkn].items():
                        dct[stemmed_tkn][type_] += freq
                except:
                    dct[stemmed_tkn] = old_dct[tkn]

    @classmethod
    def dump_json(cls, file_path: pl.Path, out_dict: dict) -> None:
        """Dump dictionary to json."""
        json_words = json.dumps(out_dict, indent=4)
        with open(file_path, "w") as outfile:
            outfile.write(json_words)