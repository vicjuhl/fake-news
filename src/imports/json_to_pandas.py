import pathlib as pl
import numpy as np
import pandas as pd
import json
from utils.functions import add_tuples


def json_to_pd(file_path : str = "data_files/words/included_words10k.json") -> pd.DataFrame:
    """Take a json file location as argument and convert it to a pandas dataframe.
     The dataframe is filtered to only show the columns: word, fake, real.
     
     - Argument: File location is relative to the fake-news folder"""

    # file reference for dataframe
    json_file_path = pl.Path(__file__).resolve().parent.parent.parent / file_path

    # creating dataframe by reading json file directly
    df = pd.read_json(json_file_path, orient="index")

    # filtering for fake and reliable and replacing NaN with [0,0]
    df = df.filter(items=['fake', 'reliable'], axis=1)
    df = df.rename(columns={'reliable':'real'})
    df = df.applymap(lambda x: [0,0] if x is np.nan else x)
    df['freq'] = df.apply(
        lambda row: [row['real'][0] + row['fake'][0], row['real'][1] + row['fake'][1]], axis=1
        ) # adding freq entry column
    
    def sort_by_second_elm(lst): # helper function for sorting by second element in lst
        return lst[1]

    df = df.sort_values(by='freq', key=lambda x: -x.map(sort_by_second_elm)) # sort by total frequency
    
    return df