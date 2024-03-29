import pandas as pd
import os 
import pathlib as pl
import math
from typing import Optional

def add_tuples(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    """Add two two-element integer tuples elementwise."""
    a1, a2 = a
    b1, b2 = b
    return (a1 + b1, a2 + b2)

def to_binary(pred: int) -> Optional[str]:
    return "fake" if pred < 0 else "reliable" if pred >= 0 else None
    
def entropy(word_dict: dict[str,int], length : int):
    """Calculate the entropy of a text."""
    entropy = -sum(freq/length * math.log2(freq/length) for freq in word_dict.values()) # # entropy using formula: text using the formula: - sum(freq/total_chars * log2(freq/total_chars))
    return entropy

def add_features_df(df : pd.DataFrame, feature_name : str) -> pd.DataFrame:
    """Add a key to a dictionary if it doesn't exist yet."""
    df['words'] = df.apply(lambda row: {**row['words'], feature_name : row[f'{feature_name}']}, axis=1) # add feature to words_dict
    return df

def create_dict_MetaModel(row):
    return {**row.to_dict()}

def del_csv(path : pl.Path):
    """Delete a csv file."""
    if os.path.exists(path):
        os.remove(path)
    else:
        print("The file does not exist")

'''
def draw_tree(model, df, size=10, ratio=0.6, precision=2, **kwargs):
    s= export_graphviz(t, out_file=None, feature_names=df.columns, filled=True, rounded=True,
                      special_characters=True, rotate=False, precision=precision, **kwargs)
    return graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s))
'''