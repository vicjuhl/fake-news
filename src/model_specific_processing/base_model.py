import pandas as pd
from abc import ABC, abstractmethod
import pathlib as pl
from typing import Optional
import json
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score # type: ignore

class BaseModel(ABC):
    '''Abstract class for models'''
    def __init__(
        self,
        params: dict,
        training_sets: dict,
        val_set: int,
        models_dir: pl.Path,
        t_session: str,
        name: str,
    ) -> None:  # 1 as default value for val_set
        self._session_dir = models_dir / f"{name}/{name}_{t_session}/"
        self._session_dir.mkdir(parents=True, exist_ok=True) # Create dest folder if it does not exist
        self._model_path = self._session_dir / f"{name}_model.csv"
        self._params = params
        self._training_sets = training_sets
        self._val_set = val_set
        self._name = name
        self._t_session = t_session
        self._data_path =  pl.Path(__file__).parent.parent.resolve() / "data_files/"
        self._preds: Optional[pd.DataFrame] = None
        self.dump_metadata()

    def dump_metadata(self) -> None:
        """Dump json file with session metadata."""
        metadata = {
            "valset_used": self._val_set,
            "session_timestamp": self._t_session,
            "params": self._params,
        }
        json_data = json.dumps(metadata, indent=4)
        with open(self._session_dir / "metadata.json", "w") as outfile:
            outfile.write(json_data)
    
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