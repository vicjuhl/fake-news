import pandas as pd
import math
import os 
from collections import Counter


def add_tuples(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    """Add two two-element integer tuples elementwise."""
    a1, a2 = a
    b1, b2 = b
    return (a1 + b1, a2 + b2)

def to_binary(type_: str) -> int:
    if type_ not in ["fake", "reliable"]:
        raise ValueError("Type must either be fake or reliable.")
    return -1 if type_ == "fake" else 1
    
def df_type_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Converts the 'type' column of a dataframe to a binary column."""
    df['type_binary'] = df['type'].apply(lambda x: to_binary(x))
    return df

def entropy(word_dict: dict[str,int], len : int):
    """Calculate the entropy of a text."""
    entropy = -sum(freq/len * math.log2(freq/len) for freq in word_dict.values()) # # entropy using formula: text using the formula: - sum(freq/total_chars * log2(freq/total_chars))
    return entropy

def add_features_df(df : pd.DataFrame, feature_name : str) -> pd.DataFrame:
    """Add a key to a dictionary if it doesn't exist yet."""
    df['words_dict'] = df.apply(lambda row: {**row['words_dict'], feature_name : row[f'{feature_name}']}, axis=1) # add feature to words_dict
    return df

def create_dict_MetaModel(row):
    return {**row.to_dict(), 'mm_inference': row.drop('type').to_dict()}

def del_csv(path : str):
    """Delete a csv file."""
    if os.path.exists(path):
        os.remove(path)
    else:
        print("The file does not exist")
