
import pandas as pd
import pickle
import pathlib as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import time
from abstract_class import abstract_model

class pa_classifier(abstract_model):
    '''PassiveAggressiveClassifier model'''
    def __init__(self, val_set: int, name:str) -> None:
        super().__init__( val_set)
        self._model = PassiveAggressiveClassifier(max_iter=50)
        self._model_dumped = False
        self._vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        self._path = None # path to model
        self.name = name
        self._val_set = val_set
        
    def data_prep(self, **kwargs) -> pd.DataFrame:
        '''Prepares the data for training'''
        t0 = time.time()
        self._train_data = kwargs['data'] # getting the data
        print(f'time to prepare data {time.time() - t0} seconds')
      
    def train(self, **kwargs) -> None:
        '''Trains a PassiveAggressiveClassifier model on the training data'''
        t0 = time.time()
        x_train = self._train_data['shortened']
        x_test = self._train_data['type']
        x_train_vec = self._vectorizer.fit_transform(x_train)
        self._model.fit(x_train_vec, x_test)
        print(f'time to training {time.time() - t0} seconds')
            
    def dump(self, to_path:str) -> None:
        '''Dumps the model to a pickle file'''
        self._path = pl.Path(f'../{to_path}/{self._val_set}{self.name}.pkl')
        with open(self._path, 'wb') as f:
            pickle.dump(self._model , f)
        self._model_dumped = True
        print(f'model dumped to {self._path}')
   
    def infer(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Makes predictions on a dataframe'''
        t0 = time.time()
        path = pl.Path(f'{self._path}')
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f) 
            
            df[f'preds_from_{self.name}'] = model.predict(self._vectorizer.transform(df['shortened'])) # adding predictions as a column
            return df
        except FileNotFoundError:
            print('cannot make inference without a trained model')    
        
        print(f'time to inference {time.time() - t0} seconds')
        
    def evaluate(self, df: pd.DataFrame) -> None: # assuming predictions are in a column called 'preds..''
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
               
           