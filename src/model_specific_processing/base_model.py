import pathlib as pl
import pandas as pd
from abc import ABC, abstractmethod
import pathlib as pl
from typing import Optional, Any
import json
from sklearn.metrics import f1_score, balanced_accuracy_score # type: ignore
from sklearn.utils.validation import check_is_fitted # type: ignore
from sklearn.exceptions import NotFittedError # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pickle

from utils.functions import to_binary # type: ignore


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
        file_format: str,
    ) -> None:  # 1 as default value for val_set
        self._session_dir = models_dir / f"{name}/{name}_{t_session}/"
        self._evaluation_dir = self._session_dir / "evaluation/"
        # Create dest folder with sub-folders if it does not exist.
        self._evaluation_dir.mkdir(parents=True, exist_ok=True)
        self._model_path = self._session_dir / f"model.{file_format}"
        self._params = params
        self._training_sets = training_sets
        self._val_set = val_set
        self._name = name
        self._t_session = t_session
        self._data_path =  pl.Path(__file__).parent.parent.resolve() / "data_files/"
        self._preds: Optional[pd.DataFrame] = None
        self._metamodel_path = models_dir / "meta_model"
        self._metamodel_path.mkdir(parents=True, exist_ok=True)
        self._metamodel_train_path =  self._metamodel_path / "metamodel_train.csv"
        self._metamodel_inference_path =  self._metamodel_path / "metamodel_inference.csv"
        self._preds_mm_training = pd.DataFrame()
        self.dump_metadata()
        self.filetype = file_format
        self._savedmodel_path = pl.Path(__file__).parent.parent.parent.resolve() / "model_files_shared" / "saved_models/"

    def dump_metadata(self) -> None:
        """Dump json file with session metadata."""
        metadata = {
            "valset_used": self._val_set,
            "session_timestamp": self._t_session,
            "params": self._params,
        }
        json_data = json.dumps(metadata, indent=4)
        with open(self._session_dir / f"metadata.json", "w") as outfile:
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

    def load(self) -> None:
        try:
            if self.filetype == "pkl":
                saved_model = pickle.load(open(
                    self._savedmodel_path / self._name / ("model" + "." + self.filetype),
                    'rb'
                ))
            elif self.filetype == "csv": # Simple model
                saved_model = pd.read_csv(
                    self._savedmodel_path / self._name / ("model" + "." + self.filetype),
                    index_col=0
                )
            self.set_model(saved_model)
        except:
            raise Exception("Exception load failed: modelfile not found")
        
    @abstractmethod
    def set_model(self, model: Any) -> None:
        pass
    
    def evaluate(self) -> None:
        '''Evaluates the model on a dataframe'''
        def try_divide(a, b):
            try:
                return a/(a+b)
            except ZeroDivisionError:
                return None

        _preds = self._preds
        if _preds is None:
            print('cannot evaluate without predictions')
            return
        preds = _preds[f'preds_{self._name}'].apply(to_binary) # predictions
        labels = _preds['type'] # correct answers 
        
        #counts results, assuming fake is the positve case 
        tp = 0  # true and fake
        fp = 0  # false and fake
        tn = 0  # true and reliable
        fn = 0  # false and reliable
        for pred, lab in zip(preds, labels):
            if pred == "fake":
                if lab == "fake":
                    tp +=1
                else:
                    fp +=1
            else: #ie. pred == reliable
                if lab == "reliable":
                    tn +=1
                else:
                    fn +=1
        #stats
        total_preds = tp + tn + fp + fn
        accuracy = (tp + tn)/total_preds
        balanced_accuracy = balanced_accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        precision = try_divide(tp, fp)
        npv = try_divide(tn, fn)#reverse precision
        recall = try_divide(tp, fn)
        tnr = try_divide(tn, fp) #reverse recall
        confusion_matrix = [[tp/total_preds, fp/total_preds], [fn/total_preds, tn/total_preds]]

        #makes dict out off stats
        eval_dict = { 
            "nPredictions": total_preds,
            "F1 Score": f1,
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_accuracy,
            "Precision": precision,
            "NPV": npv,
            "Recall": recall,
            "TNR": tnr,
            "Confusion Matrix": confusion_matrix,
        } 
        print(eval_dict)
        
        #dump stats to json
        json_eval = json.dumps(eval_dict, indent=4)
        print(json_eval)
        with open(self._evaluation_dir / "eval.json", "w") as outfile:
            outfile.write(json_eval)

        # Confusion matrix plot 
        fig, ax = plt.subplots()
        table = ax.matshow(confusion_matrix, cmap ='Blues')
   
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Fake', 'Reliable'])
        ax.set_yticklabels(['Fake', 'Reliable'])
        ax.set_xlabel('True label')
        ax.set_ylabel('Predicted label')

        # Add the values to the table
        for i in range(2):
           for j in range(2):
               text = f"{round(confusion_matrix[i][j]*100, 2)}%"
               ax.text(j, i, text, va='center', ha='center', fontsize=11)

        # dump to png
        fig.savefig((self._evaluation_dir / 'ConfusionMatrix.png'))