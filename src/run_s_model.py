
from preprocessing import noise_removal as nr

from model_specific_processing import simple_model as sm

import pandas as pd


df = pd.read_csv('data_files/processed_csv/reduced_corpus_csv.csv')

def run_model_on_df(df: pd.DataFrame):
    preprocessed_df = nr.preprocess(df)
    simple_model = sm.build(preprocessed_df, True)
    return simple_model
    

run_model_on_df(df)



