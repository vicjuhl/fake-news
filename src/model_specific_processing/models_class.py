import pandas as pd
import pickle
import pathlib as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod, method

class abstract_model(ABC):
    '''Abstract class for models'''
    def __init__(self, val_set: int) -> None:
        self._val_set = val_set
        self.models = {}
      
    @abstractmethod
    def data_prep(**kwargs) -> pd.DataFrame: # should it exist?
        pass  
    
    @abstractmethod
    def train(self, **kwargs) -> None:
        pass
    
    @abstractmethod
    def dump(self, to_path:str) -> None:
        pass
    
    @abstractmethod
    def load(self, from_path:str) -> None:
        pass

    @abstractmethod
    def infer(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> None:
        pass
    
    @method 
    def add_object(self, obj_name, obj):
        self.objects[obj_name] = obj
    
class pa_classifier(abstract_model):
    '''PassiveAggressiveClassifier model'''
    def __init__(self, val_set: int, name:str) -> None:
        super().__init__(self, val_set)
        self._model = PassiveAggressiveClassifier(max_iter=50)
        self._model_dumped = False
        self._vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        self._path = None # path to model
        self.name = name
        
    def data_prep(self, **kwargs) -> pd.DataFrame:
        '''Prepares the data for training'''
        self._train_data = kwargs['data'] # getting the data
      
    def train(self, **kwargs) -> None:
        '''Trains a PassiveAggressiveClassifier model on the training data'''
        self._train_data = kwargs['data'] # getting the data
        x_train = self._train_data['shortened']
        x_test = self._train_data['type']
        self._vectorizer.fit_transform(x_train)
        self._model.fit(self._vectorizer, x_test)
            
    def dump(self, to_path:str) -> None:
        '''Dumps the model to a pickle file'''
        with open(f'to_path/{self.name}.pkl', 'wb') as f:
            pickle.dump(self._model , f)
        self._model_dumped = True
        self._path = pl.Path(f'to_path/{self.name}.pkl')
   
    def infer(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Makes predictions on a dataframe'''
        self._path = pl.Path(f'self._path/{self.name}.pkl')
        try:
            with open(self._path, 'rb') as f:
                model = pickle.load(f) 
            
            df[f'preds_from_{self.name}'] = model.predict(self._vectorizer.transform(df['shortened'])) # adding predictions as a column
            return df
        except FileNotFoundError:
            print('cannot make inference without a trained model')    
        
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
        
        
           