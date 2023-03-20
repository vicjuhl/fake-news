import pandas as pd
from abc import ABC, abstractmethod 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import time 
import pathlib as pl
from typing import Optional

class BaseModel(ABC):
    '''Abstract class for models'''
    def __init__(
        self,
        training_sets: dict,
        val_set: int,
    ) -> None:  # 1 as default value for val_set
        self._training_sets = training_sets
        self._val_set = val_set
        self._preds: Optional[pd.DataFrame] = None
      
    # @abstractmethod
    # def data_prep(**kwargs) -> pd.DataFrame: # should it exist?
    #     pass  
    
    @abstractmethod
    def train(self, **kwargs) -> None:
        pass
    
    @abstractmethod
    def dump_model(self, to_path:str) -> None:
        pass
    
    @abstractmethod
    def infer(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    # @abstractmethod
    # def dump_inference(self, to_path:str, df : pd.DataFrame) -> None:
    #     pass
    
    
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
        
        cm = confusion_matrix(test_df["type"], preds)
        tn, fp, fn, tp = cm.ravel()
        print('Confusion matrix:')
        print(f'True positives: {tp}')
        print(f'True negatives: {tn}')
        print(f'False positives: {fp}')
        print(f'False negatives: {fn}')