import pandas as pd
from abc import ABC, abstractmethod 
from typing import Optional
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score # type: ignore

class BaseModel(ABC):
    '''Abstract class for models'''
    def __init__(
        self,
        training_sets: dict,
        val_set: int,
        name: str,
    ) -> None:  # 1 as default value for val_set
        self._training_sets = training_sets
        self._val_set = val_set
        self._preds: Optional[pd.DataFrame] = None
        self._name = name
    
    @abstractmethod
    def train(self, **kwargs) -> None:
        pass
    
    @abstractmethod
    def dump_model(self) -> None:
        pass
    
    @abstractmethod
    def infer(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    # @abstractmethod
    # def dump_inference(self, to_path:str, df : pd.DataFrame) -> None: TODO
    #     pass
    
    
    def evaluate(self) -> None:
        '''Evaluates the model on a dataframe'''
        _preds = self._preds
        if _preds is None:
            print('cannot evaluate without predictions')
            return
        
        # try:
        preds = _preds[f'preds_from_{self._name}']
        print(f'the prediction column: preds_from_{self._name}')
        # except KeyError:
        
        print("here are the stats for the model:")
        print(f'Accuracy: {accuracy_score(_preds["type"], preds)}') # type is the column with labels
        print(f'Precision: {precision_score(_preds["type"], preds, average="weighted")}')
        print(f'Recall: {recall_score(_preds["type"], preds, average="weighted")}')
        print(f'F1: {f1_score(_preds["type"], preds, average="weighted")}')
        print(f'Confusion matrix: {confusion_matrix(_preds["type"], preds)}') 
        
        cm = confusion_matrix(_preds["type"], preds)
        tn, fp, fn, tp = cm.ravel()
        print('Confusion matrix:')
        print(f'True positives: {tp}')
        print(f'True negatives: {tn}')
        print(f'False positives: {fp}')
        print(f'False negatives: {fn}')