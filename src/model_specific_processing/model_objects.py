import pandas as pd
from obj_simple_model import simple_model
#from preprocessing.noise_removal import preprocess_string 
models_dict = {}

df = pd.read_csv('..\data_files\processed_csv\summarized_corpus_valset2_full.csv')
simple_model = simple_model(2, 'simple_model1')

simple_model.data_prep(data = df)
simple_model.train(data = df)
simple_model.dump()
simple_model.infer(df)
simple_model.evaluate(df)

print(models_dict)

