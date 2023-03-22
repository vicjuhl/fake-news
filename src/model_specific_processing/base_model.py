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
        self._session_dir.mkdir(parents=True, exist_ok=True) # Create dest folder if it does not exist
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
        
        # try:
        preds = _preds[f'preds_from_{self._name}']
        labels = _preds['type']
        print(f'the prediction column: preds_from_{self._name}')
        # except KeyError:
        
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
        #precison for both fake and reliable
        fake_precision =true_fake/(true_fake + false_fake)
        reliable_precision =true_reliable/(true_reliable + false_reliable)
        #recall
        fake_recall =true_fake/(true_fake + false_reliable)
        reliable_recall =true_reliable/(true_reliable + false_fake)


        # Confusion matrix
        confusion_matrix = [[round(true_fake/total_preds, 2), round(false_fake/total_preds, 2)], [round(false_reliable/total_preds, 2), round(true_reliable/total_preds, 2)]]

        # Create a figure and axes object
        fig, ax = plt.subplots()

        # Create the table using matshow
        table = ax.matshow(confusion_matrix, cmap ='Blues')

        # Set the x and y tick labels
        ax.set_xticklabels(['', 'Fake', 'Reliable'])
        ax.set_yticklabels(['', 'Fake', 'Reliable'])

        # Add labels for the values in the table
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(confusion_matrix[i][j]), va='center', ha='center')

        # Add a title
        ax.set_title('Confusion Matrix')

        # Save the figure to a file
        fig.savefig('my_figure.png')
        plt.show()


        print("here are the stats for the model:")
        print(f'Accuracy: {accuracy}')
        print(f'Balanced Accuracy: {balanced_accuracy}')
        print(f'Fake Precision: {fake_precision}')
        print(f'Fake Recall: {fake_recall}')
        print(f'Reliable Precision: {reliable_precision}')
        print(f'Reliable Recall: {reliable_recall}')
        print(f'F1: {f1_score(_preds["type"], preds, average="weighted")}')
        print(f'Confusion matrix: {confusion_matrix}') 
        

        



