import pandas as pd
from abc import ABC, abstractmethod 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import time 
class abstract_model(ABC):
    '''Abstract class for models'''
    def __init__(self, val_set: int) -> None:  # 1 as default value for val_set
        self._val_set = val_set
      
    @abstractmethod
    def data_prep(**kwargs) -> pd.DataFrame: # should it exist?
        pass  
    
    @abstractmethod
    def train(self, **kwargs) -> None:
        pass
    
    @abstractmethod
    def dump(self, to_path:str) -> None:
        pass
    
    '''
    @abstractmethod
    def load(self, from_path:str) -> None:
        pass
    '''
    
    @abstractmethod
    def infer(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def evaluate(self, test_df: pd.DataFrame) -> None:
        '''Evaluates the model on a dataframe'''
        try:
            preds = test_df[f'preds_from_{self.name}']
        except KeyError:
            print('cannot evaluate without predictions')        
        
        print("here are the stats for the model:")
        print(f'Accuracy: {accuracy_score(test_df["type"], preds)}') # type is the column with labels
        print(f'Precision: {precision_score(test_df["type"], preds, average="weighted")}')
        print(f'Recall: {recall_score(test_df["type"], preds, average="weighted")}')
        print(f'F1: {f1_score(test_df["type"], preds, average="weighted")}')
        print(f'Confusion matrix: {confusion_matrix(test_df["type"], preds)}') 
        
class base_model(abstract_model):
    '''Base model, with evaluation standard'''
    def __init__(self, val_set: int, name:str ) -> None:
        super().__init__(val_set)
        self._name = name
        
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