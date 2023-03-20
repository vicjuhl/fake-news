import pandas as pd
from model_specific_processing.obj_simple_model import Simple_Model
import pathlib as pl
#from preprocessing.noise_removal import preprocess_string 
models_dict = {}

data_path = pl.Path(__file__).parent.parent.resolve() / "data_files/"
file_path = data_path / 'words/stop_words_removed_valset2.json'

df = pd.read_csv(data_path / 'processed_csv/summarized_corpus_valset2_100k.csv')
df = df[:100]
sm = Simple_Model(2, 'simple_model1')


sm.data_prep(path = file_path)
sm.train()
sm.dump()
sm.infer(df)
sm.evaluate(df)
models_dict[sm.name] = sm


