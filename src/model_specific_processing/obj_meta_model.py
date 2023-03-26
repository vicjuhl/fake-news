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
            
            print(train_data.columns)
            labels = train_data['type'] # strings
            
            train_data.drop(['id', 'type'], axis = 1, inplace=True) # should not be used for training because of information polution      
            train_data = train_data.applymap(lambda x: 1 if x == 'reliable' else 0)
            train_data['dict'] =  train_data.apply(create_dict_MetaModel, axis=1)
            print(train_data['dict'])
            
            train_data_vec = self._vectorizer.fit_transform(train_data['dict'].to_list())
            self._model.fit(train_data_vec, labels)    
        except FileNotFoundError or pd.errors.EmptyDataError:
            print('metamodel cannot be trained, empty csv')   
             
    def dump_for_mm_training(self):
        pass   
    
    def dump_model(self) -> None:
        '''Dumps the model to a pickle file'''
        with open(self._model_path, 'wb') as f:
            pickle.dump(self._model , f)
        print(f'model dumped to {self._model_path}')
        
    def infer(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Infer the model on the given dataframe'''
        # load model
        ''''''
        try:   
            if self._model is None:
                    with open(self._model_path, 'rb') as f:
                        model = pickle.load(f)
            else:
                model = self._model
            
            print('not ready yet')
            '''           
            labels = train_data['type']
            
            train_data.drop('id', axis = 1, inplace=True)     
            train_data.drop('type', axis = 1,  inplace=True)    
            
            df.drop('id', axis=1, inplace=True) # droppping id column
            df = df.applymap(lambda x: 1 if x == 'reliable' and x != 'type' else 0 if x == 'fake' and x != 'type' else x)
            print(df.head(5))
            df['inference_column'] = df.apply(create_dict_MetaModel, axis=1)  
            self._preds = df[['id', 'type', 'split']].copy()
            
            self._preds[f'preds_{self._name}'] = model.predict(
                self._vectorizer.transform(df['inference_column']))
            '''
        
        
        except FileNotFoundError:
            print('Cannot make inference without a trained model')    

        