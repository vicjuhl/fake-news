import pathlib as pl
import numpy as np
import pandas as pd
import os

import json


df = pd.read_csv("data_files/corpus/splits.csv", nrows=1000)
print(df)

'''
with open('../data_files/stop_words_removed.json', 'r') as f:
    data = json.load(f)
    
print(type(data))

df = pd.DataFrame.from_dict(data, orient='index')
print(df.head(10))
'''
