import pandas as pd
from abc import ABC, abstractmethod
import pathlib as pl
from typing import Optional
import json
from sklearn.metrics import f1_score, balanced_accuracy_score # type: ignore
import matplotlib.pyplot as plt

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
        file_type: str,
    ) -> None:  # 1 as default value for val_set
        self._session_dir = models_dir / f"{name}/{name}_{t_session}/"
        self._evaluation_dir = self._session_dir / "evaluation/"
        # Create dest folder with sub-folders if it does not exist.
        self._evaluation_dir.mkdir(parents=True, exist_ok=True)
        self._model_path = self._session_dir / f"model.{file_type}"
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
    
    # @abstractmethod
    # def dump_inference(self, to_path:str, df : pd.DataFrame) -> None: TODO
    #     pass
    
    def evaluate(self) -> None:
        '''Evaluates the model on a dataframe'''
        _preds = self._preds
        if _preds is None:
            print('cannot evaluate without predictions')
            return
        preds = _preds[f'preds_from_{self._name}'] #predictions
        labels = _preds['type'] #correct anwsers
        
        #counts results
        true_fake = 0
        false_fake = 0
        true_reliable = 0
        false_reliable = 0
        for pred, lab in zip(preds, labels):
            if pred == "fake":
                if lab == "fake":
                    true_fake +=1
                else:
                    false_fake +=1
            else: #ie. pred == reliable
                if lab == "reliable":
                    true_reliable +=1
                else:
                    false_reliable +=1
        total_preds = true_fake + true_reliable + false_fake + false_reliable

        #stats
        accuracy = (true_fake + true_reliable)/total_preds
        balanced_accuracy = balanced_accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        #precison for both fake and reliable
        fake_precision =true_fake/(true_fake + false_fake)
        reliable_precision =true_reliable/(true_reliable + false_reliable)
        #recall
        fake_recall =true_fake/(true_fake + false_reliable)
        reliable_recall =true_reliable/(true_reliable + false_fake)
        # Confusion matrix
        confusion_matrix = [[round(true_fake/total_preds, 2), round(false_fake/total_preds, 2)], [round(false_reliable/total_preds, 2), round(true_reliable/total_preds, 2)]]

        #makes dict with stats
        
        eval_dict = { 
            "nPredictions": total_preds,
            "F1 Score": f1,
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_accuracy,
            "Fake Precision": fake_precision,
            "Fake Recall": fake_recall,
            "Reliable Precision": reliable_precision,
            "Reliable Recall": reliable_recall,
            "Confusion Matrix": confusion_matrix,
        } 

        #dump stats to json
        json_eval = json.dumps(eval_dict, indent=4)
        with open(self._evaluation_dir, "w") as outfile:
            outfile.write(json_eval)

        # Confusion matrix plot 
        fig, ax = plt.subplots()
        table = ax.matshow(confusion_matrix, cmap ='Blues')
        ax.set_xticklabels(['', 'Fake', 'Reliable'])
        ax.set_yticklabels(['', 'Fake', 'Reliable'])

        # Add the values to the table
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(confusion_matrix[i][j]), va='center', ha='center')
        ax.set_title('Confusion Matrix')

        #dump to png
        fig.savefig((self._evaluation_dir / 'figure.png'))

        

        



