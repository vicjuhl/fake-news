import pathlib as pl
import pandas as pd
from abc import ABC, abstractmethod
import pathlib as pl
from typing import Optional
import json
import os
from sklearn.metrics import f1_score, balanced_accuracy_score # type: ignore
import matplotlib.pyplot as plt
from utils.functions import to_binary
from utils.functions import del_csv # type: ignore

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
        self._metamodel_path = models_dir / "metamodel"
        self._metamodel_path.mkdir(parents=True, exist_ok=True)
        self._metamodel_train_path =  self._metamodel_path / "metamodel_train.csv"
        self._metamodel_inference_path =  self._metamodel_path / "metamodel_inference.csv"
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
    
    def dump_for_mm_training(self):
        '''Dumps predictions to a csv for metamodel to train on'''
        print('generating training data for metamodel, dumping predictions')
        
        try:
            # load existing metamodel CSV file into a DataFrame
            mm_df = pd.read_csv(self._metamodel_train_path)
        except Exception as e:
            print("Not loading csv: ", e)
            mm_df = pd.DataFrame({'id': self._preds.id, 'type': self._preds.type})
            
        col_name = f'preds_{self._name}'
        try:
            # add new predictions as a new column to existing DataFrame
            col_name = f'preds_{self._name}'
            if col_name not in self._preds:
                print(f'no predictions to dump for {self._name}')

            if col_name in mm_df.columns:
                mm_df = mm_df.drop(col_name, axis=1) # dropping column if it already exists 
            
            if 'preds_simple_cont' in mm_df.columns:
                mm_df = mm_df.drop('preds_simple_cont', axis=1) # dropping column if it already exists
            
            mm_df = pd.merge(
                mm_df,
                self._preds.drop(["type", "split"], axis=1), # problem, adding other columns than just preds!!
                on="id",
                how="left",
                suffixes=("_l", "_r")
            )
            # save updated DataFrame to metamodel CSV file
            mm_df.to_csv(self._metamodel_train_path, mode="w", index=False)
        except Exception as e:
            print(f'Something went wrong adding predictions: {e}')
    
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
        confusion_matrix = [[round(tp/total_preds, 2), round(fp/total_preds, 2)], [round(fn/total_preds, 2), round(tn/total_preds, 2)]]

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

        # Add the values to the table
        for i in range(2):
            for j in range(2):
                text = f"{round(confusion_matrix[i][j]*100, 2)}%"
                ax.text(j, i, text, va='center', ha='center', fontsize=11)

        #dump to png
        fig.savefig((self._evaluation_dir / 'ConfusionMatrix.png'))
      