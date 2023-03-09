import pathlib as pl
import numpy as np
import pandas as pd
import os
import json

def json_to_pd(file_path : str = "data_files/words/included_words.json") -> pd.DataFrame:
    """Takes a json file location as argument and converts it to a pandas dataframe.
     Filtered to only show columns: word, fake, real.
     
     - Argument: File location is relative to the fake-news folder"""

    # file reference for dataframe
    json_file_path = pl.Path(os.path.abspath('')).parent.resolve() / file_path

    # open and read the json file
    with open(json_file_path) as json_file:
        json_dict = json.load(json_file)

    # creating dataframe
    df = pd.DataFrame.from_dict({(word): json_dict[word]
                            for word in json_dict.keys()},
                        orient='index')

    # filtering for fake and reliable and replacing NaN with [0,0]
    df = df.filter(items=['fake', 'reliable'], axis=1)
    df = df.rename(columns={'reliable':'real'})
    df = df.applymap(lambda x: [0,0] if x is np.nan else x)

    return df

