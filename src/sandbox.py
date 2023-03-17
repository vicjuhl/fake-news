import pathlib as pl
import numpy as np
import pandas as pd
import os

import json


with open('../data_files/stop_words_removed.json', 'r') as f:
    json_data = f.read()
data = json.loads(json_data)

df = pd.DataFrame.from_dict(data, orient='index')
print(df.head())

'''
with open('../data_files/stop_words_removed.json', 'r') as f:
    data = json.load(f)
    
print(type(data))

df = pd.DataFrame.from_dict(data, orient='index')
print(df.head(10))
'''
