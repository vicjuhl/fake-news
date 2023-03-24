from sklearn.linear_model import LogisticRegression # type: ignore
import pathlib as pl 
from typing import Optional
import pandas as pd
from sklearn.feature_extraction import DictVectorizer # type: ignore
import pickle
import os 

from utils.functions import create_dict_MetaModel
from model_specific_processing.base_model import BaseModel  # type: ignore
from utils.functions import del_csv
class MetaModel(BaseModel):
    def __init__(
         self,
        params: dict,
        training_sets: dict,
        val_set: int,
        models_dir: pl.Path,
        t_session: str,
        name : str = "meta_model",
        model_format : str = "pkl"
    ) -> None:
        super().__init__(params, training_sets, val_set, models_dir, t_session, name, model_format)
        self._model = LogisticRegression(max_iter=1000, n_jobs=-1)
        self._vectorizer = DictVectorizer()
    def train(self) -> None:
        '''Train the model on the given training sets'''
        try:            
            train_data = pd.read_csv(self._metamodel_train_path)
            print(train_data)
            train_data =  train_data.apply(create_dict_MetaModel, axis=1)
            print(train_data)
            train_data_vec = self._vectorizer.fit_transform(train_data).to_list()
            self._model.fit(train_data_vec, train_data['type'])    
            print('metamodel trained, now wiping values') 
            empty_df = pd.DataFrame()
            empty_df.to_csv(self._metamodel_train_path, index=False) # empty the metamodel csv     
        except FileNotFoundError or pd.errors.EmptyDataError:
            print('metamodel cannot be trained, empty csv')        
            
    def dump_model(self) -> None:
        '''Dumps the model to a pickle file'''
        with open(self._model_path, 'wb') as f:
            pickle.dump(self._model , f)
        print(f'model dumped to {self._model_path}')
        
    def infer(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Infer the model on the given dataframe'''
        # load model
        with open(self._model_path, 'rb') as f:
            self._model = pickle.load(f)
        try:
            df = pd.read_csv(self._metamodel_inference_path) # inference on metamodel csv
            print('wiping predictions from csv')
            del_csv(self._metamodel_inference_path)
            # infer
            df['inference_column'] = df.apply(create_dict_MetaModel, axis=1).to_list()    
            df[f'preds_from_{self._name}'] = self._model.predict(df['inference_column'])
            df.drop(columns=['inference_column'], inplace=True)   
        except FileNotFoundError:
            print('no metamodel csv cannot do inference, run at least one model first')
        except pd.errors.EmptyDataError:
            print('no metamodel csv cannot do inference, run at least one model first')
        self._preds = df