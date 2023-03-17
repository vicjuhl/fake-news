import pathlib as pl
import numpy as np
import pandas as pd
import json

def json_to_pd(file_path : str = "data_files/words/included_words.json") -> pd.DataFrame:
    """Take a json file location as argument and convert it to a pandas dataframe.
     The dataframe is filtered to only show the columns: word, fake, real.
     
     - Argument: File location is relative to the fake-news folder"""

    # file reference for dataframe
    json_file_path = pl.Path(__file__).resolve().parent.parent.parent / file_path

    # creating dataframe by reading json file directly
    df = pd.read_json(json_file_path, orient="index")

    # filtering for fake and reliable and replacing NaN with [0,0]
    df = df.filter(items=['fake', 'reliable'], axis=1)
    #df = df.rename(columns={'reliable':'real'})
    df = df.applymap(lambda x: [0,0] if x is np.nan else x)

    return df