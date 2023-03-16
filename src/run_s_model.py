from preprocessing import noise_removal as nr
from model_specific_processing import simple_model as sm
import os
import pandas as pd
import pathlib as pl
import numpy as np
import json


# file reference for dataframe
json_file_path = pl.Path(os.path.abspath('')).parent.resolve() / "data_files/100k/included_words_100k.json"

with open(json_file_path) as json_file:
    json_dict = json.load(json_file)

# filtering for fake and reliable and replacing NaN with [0,0]
df_json = pd.DataFrame.from_dict({(word): json_dict[word]
                           for word in json_dict.keys()},
                       orient='index')
                       
df = df_json.copy()
df = df.filter(items=['fake', 'reliable'], axis=1)
df = df.applymap(lambda x: [0,0] if x is np.nan else x)



#build model
def build_model(df: pd.DataFrame):
    preprocessed_df = nr.preprocess(df)
    simple_model = sm.build_model(preprocessed_df, 50000, True)
    return simple_model


csv_file_path = pl.Path(os.path.abspath('')).parent.resolve() / "data_files/news_sample.csv"
article_df = pd.read_csv(csv_file_path)
sm.infer(article_df, build_model(df))

        

