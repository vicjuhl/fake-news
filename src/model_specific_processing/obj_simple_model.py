import pandas as pd
import pickle
import pathlib as pl
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import time
from typing import Optional
import os
from model_specific_processing.simple_model import frequency_adjustment, tf_idf, logistic_Classification_weight, create_model, save_to_csv, model_processing, classify_article, binary_classifier 
from model_specific_processing.base_model import BaseModel
import csv 
from imports.json_to_pandas import json_to_pd

class SimpleModel(BaseModel):
    '''Simple model'''
    def __init__(self, training_sets: dict, val_set: int, model_path: pl.Path) -> None:
        super().__init__(training_sets, val_set)
        self._model: Optional[pd.DataFrame] = None # a dataframe
        self.total_num_articles = None
        self._name = "simple"
        self._model_path = model_path / f"simple/{self._name}_valset{self._val_set}.csv"
        
    # def data_prep(self, **kwargs) -> pd.DataFrame:
    #     '''Prepares the data for training'''
    #     # getting the data, converting to df, adding freq column 
    #     self.total_num_articles, self._train_data = json_to_pd(kwargs['path'], return_df=True)
        
    def train(self, **kwargs) -> None:
        '''Trains a simple_model instance on the training data'''
        self._train_data = frequency_adjustment(self._train_data)
        self._train_data = tf_idf(self._train_data, self.total_num_articles)
        self._train_data = logistic_Classification_weight(self._train_data)
        
        model = create_model(self._train_data) # creating model dataframe     
        self._model = model # might not be smart to save a df in object
        
    def dump_model(self) -> None:
        '''Dumps the model to a csv file'''
        try:
            self._model.to_csv(self._model_path, index=True)
            print(f"Model saved to {self._model_path}")
        except OSError:
            # create the directory if it doesn't exist
            os.makedirs(self._model_path, exist_ok=True)
            self._model.to_csv(self._model_path, index=True)
            print(f"Model saved to {self._model_path}")
        
    def infer(self, test_df: pd.DataFrame) -> None:
        '''Makes predictions on a dataframe'''
        if self._model is None:
            self._model = pd.read_csv(self._model_path)
        # adding predictions as a column
        self._preds = test_df[f'preds_from_{self._name}'] = classify_article(test_df, self._model)
        

        
    
        
        
        