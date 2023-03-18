import simple_model as sm
import pandas as pd
import pickle
import pathlib as pl
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import time 
from abstract_class import abstract_model
import csv 

class simple_model(abstract_model):
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
        self._train_data = kwargs['data'] # getting the data
        total_num_articles = len(self._train_data) # total number of articles
        self._train_data = sm.frequency_adjustment(self._train_data)
        self._train_data = sm.tf_idf(self._train_data, total_num_articles)
        self._train_data = sm.logistic_Classification_weight(self._train_data)
        print(f'time to prepare data {time.time() - t0} seconds')
        
    def train(self, **kwargs) -> None:
        '''Trains a simple_model instance on the training data'''
        t0 = time.time()
        model = sm.create_model(self._train_data) # creating model dataframe     
        print(f'time to training {time.time() - t0} seconds')
        self._model = model # might not be smart to save a df in object
        
    default_path = '../models/simple_model_csv'
    
    def dump(self, to_path:str = default_path) -> None:
        '''Dumps the model to a csv file'''
        self._path = pl.Path(f'../{to_path}/{self._val_set}{self.name}.csv')
        sm.save_to_csv(self._model, self._path)
        self.model = None # wiping the model from the object, to save memory
        print(f'model dumped to {self._path}')
        
    def infer(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Makes predictions on a dataframe'''
        t0 = time.time()
        path = pl.Path(f'{self._path}')
        try:
            model = pd.read_csv(self._path)
            # adding predictions as a column
            df[f'preds_from_{self.name}'] = sm.classify_article(df, self._model, )
            return df
        except FileNotFoundError:
            print('cannot make inference without a trained model') 
            
        print(f'time to inference {time.time() - t0} seconds')
        
    def evaluate(self, df: pd.DataFrame) -> None:
        '''Evaluates the model on a dataframe'''
        try:
            preds = df[f'preds_from_{self.name}']
        except KeyError:
            print('cannot evaluate without predictions')        
        
        print("here are the stats for the model:")
        print(f'Accuracy: {accuracy_score(df["type"], preds)}') # type is the column with labels
        print(f'Precision: {precision_score(df["type"], preds, average="weighted")}')
        print(f'Recall: {recall_score(df["type"], preds, average="weighted")}')
        print(f'F1: {f1_score(df["type"], preds, average="weighted")}')
        print(f'Confusion matrix: {confusion_matrix(df["type"], preds)}')
        
        
    
        
        
        