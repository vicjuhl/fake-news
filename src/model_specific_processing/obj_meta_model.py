from sklearn.linear_model import LogisticRegression # type: ignore
import pathlib as pl 
from typing import Optional
import pandas as pd
from sklearn.feature_extraction import DictVectorizer # type: ignore
import pickle

from utils.functions import create_dict_MetaModel
from model_specific_processing.base_model import BaseModel  # type: ignore

class MetaModel(BaseModel):
    def __init__(self, training_sets: dict, val_set: int, model_path: pl.Path) -> None: # potentially add vectorizer, linear_model as inp
        super().__init__(training_sets , val_set, "meta_model")
        self._model = LogisticRegression(max_iter=1000)
        self._vectorizer = DictVectorizer()
    def train(self) -> None:
        # load data
        try:            
            train_data = pd.read_csv(self._metamodel_csv_path)
            print(train_data)
            train_data =  train_data.apply(create_dict_MetaModel, axis=1).to_list()
            print(train_data)
            train_data_vec = self._vectorizer.fit_transform(train_data)
            self._model.fit(train_data_vec, train_data['type'])    
            print('metamodel trained, now wiping values') 
            
            empty_df = pd.DataFrame()
            empty_df.to_csv(self._metamodel_csv_path, index=False) # empty the metamodel csv     
        except ValueError:
            print('metamodel cannot be trained')        
            
    def dump_model(self) -> None:
        '''Dumps the model to a pickle file'''
        with open(self._model_path, 'wb') as f:
            pickle.dump(self._model , f)
        print(f'model dumped to {self._model_path}')
        
    def infer(self) -> pd.DataFrame:
        '''Infer the model on the given dataframe'''
        # load model
        with open(self._model_path, 'rb') as f:
            self._model = pickle.load(f)
        try:
            df = pd.read_csv(self._metamodel_csv_path) # inference on metamodel csv
            print('wiping predictions from csv')
            empty_df = pd.DataFrame()
            empty_df.to_csv(self._metamodel_csv_path, index=False) # empty the metamodel csv
            # infer
            df['inference_column'] = df.apply(create_dict_MetaModel, axis=1).to_list()    
            df[f'preds_from_{self._name}'] = self._model.predict(df['inference_column'])
            df.drop(columns=['inference_column'], inplace=True)   
        except ValueError:
            print('no metamodel csv cannot do inference, run at least one model first')
                 
        self._preds = df