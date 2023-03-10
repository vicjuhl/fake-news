
from preprocessing import noise_removal as nr

from model_specific_processing import simple_model as sm
import os
import pandas as pd

csv_path = os.path.join(os.path.dirname(__file__), '..', 'data_files', 'processed_csv', 'reduced_corpus_csv.csv')
df = pd.read_csv(csv_path)

def run_model_on_df(df: pd.DataFrame):
    preprocessed_df = nr.preprocess(df)
    simple_model = sm.build(preprocessed_df, True)
    return simple_model
    

run_model_on_df(df)



