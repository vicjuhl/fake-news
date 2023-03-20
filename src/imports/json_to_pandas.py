import pathlib as pl
import numpy as np
import pandas as pd


def json_to_pd(val_set: int, file_name: str = "stop_words_removed") -> pd.DataFrame:
    """Take a json file location as argument and convert it to a pandas dataframe.
     The dataframe is filtered to only show the columns: word, fake, real.
     
     - Argument: File location is relative to the fake-news folder"""

    # file reference for dataframe
    json_file_path = (
        pl.Path(__file__).resolve().parent.parent.parent / "data_files/words" / f"{file_name}_valset{val_set}.json"
    )

    # creating dataframe by reading json file directly
    df = pd.read_json(json_file_path, orient="index")
    try:
        df.rename(columns={'real': 'reliable'}, inplace=True)
    except KeyError:
        print('realiable not i the column, assume real means reliable')
        # if real is not in the column, assume reliable PATCH SHOULD BE REMOVED LATER

    # filtering for fake and reliable and replacing NaN with [0,0]
    df = df.filter(items=['fake', 'reliable'], axis=1)
    
    df = df.applymap(lambda x: [0,0] if x is np.nan else x)
    try:
        df['freq'] = df.apply(
            lambda row: [row['reliable'][0] + row['fake'][0], row['reliable'][1] + row['fake'][1]], axis=1
            ) # adding freq entry column
    except KeyError:
        print('realiable not i the column, assume real means reliable')
        
        df['freq'] = df.apply(
            lambda row: [row['real'][0] + row['fake'][0], row['real'][1] + row['fake'][1]], axis=1
            ) # adding freq entry column    
    
    def get_second_elm(lst): # helper function for sorting by second element in lst
        return lst[1]

    df = df.sort_values(by='freq', key=lambda x: -x.map(get_second_elm)) # sort by total frequency
    
    return df