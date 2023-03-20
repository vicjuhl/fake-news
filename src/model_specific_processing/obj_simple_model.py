import pandas as pd
import pickle
import pathlib as pl
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import time
import os

from model_specific_processing.simple_model import frequency_adjustment, tf_idf, logistic_Classification_weight, create_model, save_to_csv, model_processing, classify_article, binary_classifier 
from model_specific_processing.abstract_class import abstract_model, base_model
import csv 
from imports.json_to_pandas import json_to_pd

class Simple_Model(abstract_model):
    '''Simple model'''
    def __init__(self, val_set: int, name:str ) -> None:
        super().__init__(val_set)
        self._model_dumped = False
        self._path = None # path to model
        self.name = name
        self._val_set = val_set
        self._train_data = None
        self._model = None # a dataframe
        
    def data_prep(self, **kwargs) -> pd.DataFrame:
        '''Prepares the data for training'''
        t0 = time.time()
        self._train_data = json_to_pd(kwargs['path']) # getting the data, converting to df, adding freq column
        total_num_articles = len(self._train_data) # total number of articles
        self._train_data = frequency_adjustment(self._train_data)
        self._train_data = tf_idf(self._train_data, total_num_articles)
        self._train_data = logistic_Classification_weight(self._train_data)
        print(f'time to prepare data {time.time() - t0} seconds')
        
    def train(self, **kwargs) -> None:
        '''Trains a simple_model instance on the training data'''
        t0 = time.time()
        model = create_model(self._train_data) # creating model dataframe     
        print(f'time to training {time.time() - t0} seconds')
        self._model = model # might not be smart to save a df in object
        
    default_path = '\model_files\simple_model_csv'
    
    
    def dump(self, to_path:str = default_path) -> None:
        '''Dumps the model to a csv file'''
        self._path = os.path.join(to_path, f'{self._val_set}{self.name}.csv')
        try:
            self._model.to_csv(self._path, index=True)
            print(f"Model saved to {self._path}")
        except OSError:
            # create the directory if it doesn't exist
            os.makedirs(to_path, exist_ok=True)
            self._model.to_csv(self._path, index=True)
            print(f"Model saved to {self._path}")
        
    def infer(self, test_df: pd.DataFrame) -> pd.DataFrame:
        '''Makes predictions on a dataframe'''
        t0 = time.time()
        path = pl.Path(f'{self._path}')
        try:
            model = pd.read_csv(self._path)
            # adding predictions as a column
            test_df[f'preds_from_{self.name}'] = classify_article(test_df, self._model, )
            return test_df
        except FileNotFoundError:
            print('cannot make inference without a trained model') 
            
        print(f'time to inference {time.time() - t0} seconds')
        

        
    
        
        
        