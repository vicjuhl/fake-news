import pathlib as pl
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
        model_path = pl.Path(__file__).parent.parent.parent.resolve() / "model_files/"
        self._metamodel_csv_path = model_path / 'metamodel_files/mmdataset.csv'
        self._metamodel_csv_infer_path = model_path / 'metamodel_files/mmdataset.csv'
    
    @abstractmethod
    def train(self, **kwargs) -> None:
        pass
    
    @abstractmethod
    def dump_model(self) -> None:
        pass
    
    @abstractmethod
    def infer(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def dump_preds(self):
        print('generating training data for metamodel, dumping predictions')
        try:
            # load existing metamodel CSV file into a DataFrame
            mm_df = pd.read_csv(self._metamodel_csv_path)
        except ValueError:
            print('no metamodel csv found, creating one')
            # create new DataFrame with 'type' column only
            mm_df = pd.DataFrame()
            mm_df['type'] = self._preds['type']
        
        try:
            print(self._preds[f'preds_from_{self._name}'])
            
            # add new predictions as a new column to existing DataFrame
            mm_df[f'preds_from_{self._name}'] = self._preds[f'preds_from_{self._name}']
        except KeyError:
            print(f'no predictions to dump for {self._name}')
        
        # save updated DataFrame to metamodel CSV file
        mm_df.to_csv(self._metamodel_csv_path, index= False, mode="w+")     
        
    def dump_inference(self):
        try:
            mm_test = pd.read_csv(self._metamodel_csv_infer_path)
        except ValueError:
            print('no metamodel csv found, creating one')
            # create new DataFrame with 'type' column only
            mm_test = pd.DataFrame()
        try:
            mm_test[f'preds_from_{self._name}'] = self._preds[f'preds_from_{self._name}']
        except AttributeError: 
            print('no inference to dump')

        mm_test.to_csv(self._metamodel_csv_infer_path,  index= False, mode="w+")
    
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