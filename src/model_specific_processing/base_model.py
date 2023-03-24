import pathlib as pl
import pandas as pd
from abc import ABC, abstractmethod
import pathlib as pl
from typing import Optional
import json
import os
from sklearn.metrics import f1_score, balanced_accuracy_score # type: ignore
#import matplotlib.pyplot as plt

from utils.functions import del_csv 
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
        models_dir
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
        self._metamodel_path = models_dir / "metamodel"
        self._metamodel_path.mkdir(parents=True, exist_ok=True)
        self._metamodel_train_path =  self._metamodel_path / "metamodel_preds.csv"
        self._metamodel_inference_path =  self._metamodel_path / "mm_inference.csv"

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
    
    def dump_preds(self):
        '''Dumps predictions to a csv for metamodel to train on'''
        print('generating training data for metamodel, dumping predictions')
        
        if not os.path.exists(self._metamodel_train_path): # if csv does not exist create it
            pd.DataFrame().to_csv(self._metamodel_train_path, index=False)        
        
        try:
            # load existing metamodel CSV file into a DataFrame
            mm_df = pd.read_csv(self._metamodel_train_path)
            del_csv(self._metamodel_train_path) # delete csv file
            
        except pd.errors.EmptyDataError:
            print('no metamodel csv found, creating one')
            mm_df = pd.DataFrame()
            mm_df['type'] = self._preds['type']
        
        try:
            # add new predictions as a new column to existing DataFrame
            mm_df[f'preds_from_{self._name}'] = self._preds[f'preds_from_{self._name}']
        except KeyError:
            print(f'no predictions to dump for {self._name}')
        
        # save updated DataFrame to metamodel CSV file
        mm_df.to_csv(self._metamodel_train_path, index= False, mode="w+")     
        
    def dump_inference(self):
        '''Dumps predictions to a csv for metamodel inference'''
        if not os.path.exists(self._metamodel_train_path): # if csv does not exist create it
            pd.DataFrame().to_csv(self._metamodel_inference_path, index=False)   
        
        try:
            mm_test = pd.read_csv(self._metamodel_inference_path)
            del_csv(self._metamodel_inference_path) # delete csv file
        except pd.errors.EmptyDataError:
            print('no metamodel csv found, creating one')
            # create new DataFrame with 'type' column only
            mm_test = pd.DataFrame()
        try:
            mm_test[f'preds_from_{self._name}'] = self._preds[f'preds_from_{self._name}']
        except pd.errors.EmptyDataError: 
            print('no inference to dump')

        mm_test.to_csv(self._metamodel_inference_path,  index= False, mode="w+")
    
    def evaluate(self) -> None:
        '''Evaluates the model on a dataframe'''
        _preds = self._preds
        if _preds is None:
            print('cannot evaluate without predictions')
            return
        preds = _preds[f'preds_from_{self._name}'] #predictions
        labels = _preds['type'] #correct anwsers
        
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
        precision =tp/(tp + fp)
        npv = tn/(tn + fn) #reverse precision
        recall =tp/(tp + fn)
        tnr =tn/(tn + fp) #reverse recall
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
        print(eval_dict)
        
        #dump stats to json
        json_eval = json.dumps(eval_dict, indent=4)
        with open(self._evaluation_dir / "eval.json", "w") as outfile:
            outfile.write(json_eval)


        # Confusion matrix plot 
        #fig, ax = plt.subplots()
        #table = ax.matshow(confusion_matrix, cmap ='Blues')
        #ax.set_xticklabels(['', 'Fake', 'Reliable'])
        #ax.set_yticklabels(['', 'Fake', 'Reliable'])

        # Add the values to the table
        #for i in range(2):
        #    for j in range(2):
        #        ax.text(j, i, str(confusion_matrix[i][j]), va='center', ha='center')
        #ax.set_title('Confusion Matrix')

        #dump to png
        #fig.savefig((self._evaluation_dir / 'ConfusionMatrix.png'))
      