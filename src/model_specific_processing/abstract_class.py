import pandas as pd
from abc import ABC, abstractmethod 

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
    
    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> None:
        pass
    