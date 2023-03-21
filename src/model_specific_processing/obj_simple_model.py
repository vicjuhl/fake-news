import pandas as pd
import pathlib as pl
from typing import Optional
from model_specific_processing.simple_model import frequency_adjustment, tf_idf, logistic_Classification_weight, create_model, classify_article
from model_specific_processing.base_model import BaseModel
class SimpleModel(BaseModel):
    '''Simple model'''
    def __init__(self, training_sets: dict, val_set: int, model_path: pl.Path) -> None:
        super().__init__(training_sets, val_set)
        self._model: Optional[pd.DataFrame] = None # a dataframe
        self._training_sets = training_sets
        self._name = "simple"
        simple_path = model_path / "simple/"
        self._data_path =  pl.Path(__file__).parent.parent.resolve() / "data_files/"
        simple_path.mkdir(parents=True, exist_ok=True) # Create dest folder if it does not exist
        self._model_path = simple_path / f"{self._name}_valset{self._val_set}.csv"
        
    def train(self) -> None:
        '''Trains a simple_model instance on the training data'''
        total_num_articles, train_data = self._training_sets["bow_simple"]
        train_data = frequency_adjustment(train_data)
        train_data = tf_idf(train_data, total_num_articles)
        train_data = logistic_Classification_weight(train_data)
        model = create_model(train_data) # creating model dataframe     
        self._model = model # might not be smart to save a df in object TODO
        
    def dump_model(self) -> None:
        '''Dumps the model to a csv file'''
        self._model.to_csv(self._model_path, index=False) 
        print(f"Model saved to {self._model_path}")
        
    def infer(self, test_df) -> None:
        '''Makes predictions on a dataframe'''
        if self._model is None:
            self._model = pd.read_csv(self._model_path) 
        test_df[f'preds_from_{self._name}'] = classify_article(test_df , self._model)         
        self._preds = test_df
        
    
        
        
        