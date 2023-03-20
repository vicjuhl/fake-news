import pathlib as pl
import numpy as np
import pandas as pd


def json_to_pd(file_path : str = "data_files/words/included_words10k.json") -> tuple[int,pd.DataFrame]:
    """Take a json file location as argument and convert it to a pandas dataframe.
     The dataframe is filtered to only show the columns: word, fake, real.
     
     - Argument: File location is relative to the fake-news folder"""

    # file reference for dataframe
    json_file_path = pl.Path(__file__).resolve().parent.parent.parent / file_path

    # creating dataframe by reading json file directly
    data = pd.read_json(json_file_path)
    n_articles, df = data["nArticles"][0], data["words"][0] # unpack data

    df.set_index(df[0]) # sets labels as indexes
    # filtering for fake and reliable and replacing NaN with [0,0]
    df = df.filter(items=['fake', 'reliable'], axis=1)
    df = df.applymap(lambda x: [0,0] if x is np.nan else x)
    df['freq'] = df.apply(
        lambda row: [row['reliable'][0] + row['fake'][0], row['reliable'][1] + row['fake'][1]], axis=1
        ) # adding freq entry column
    
    def get_second_elm(lst): # helper function for sorting by second element in lst
        return lst[1]

    df = df.sort_values(by='freq', key=lambda x: -x.map(get_second_elm)) # sort by total frequency
    
    return n_articles,df