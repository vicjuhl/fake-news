import pandas as pd
import pathlib as pl
from typing import Optional
import os
from model_specific_processing.simple_model import frequency_adjustment, tf_idf, logistic_Classification_weight, create_model, classify_article
from model_specific_processing.base_model import BaseModel

class SimpleModel(BaseModel):
    '''Simple model'''
    def __init__(self, training_sets: dict, val_set: int, model_path: pl.Path) -> None:
        super().__init__(training_sets, val_set)
        self._model: Optional[pd.DataFrame] = None # a dataframe
        self.total_num_articles = 3e6 #TODO
        self._name = "simple"
        simple_path = model_path / "simple/"
        simple_path.mkdir(parents=True, exist_ok=True) # Create dest folder if it does not exist
        self._model_path = simple_path / f"{self._name}_valset{self._val_set}.csv"
        
    def train(self, **kwargs) -> None:
        '''Trains a simple_model instance on the training data'''
        train_data = self._training_sets["bag_of_words"]
        train_data = frequency_adjustment(train_data)
        train_data = tf_idf(train_data, self.total_num_articles)
        train_data = logistic_Classification_weight(train_data)
        model = create_model(train_data) # creating model dataframe     
        self._model = model # might not be smart to save a df in object TODO
        
    def dump_model(self) -> None:
        '''Dumps the model to a csv file'''
        self._model.to_csv(self._model_path, index=False) # TODO HP?
        print(f"Model saved to {self._model_path}")
        
    def infer(self) -> None:
        '''Makes predictions on a dataframe'''
        if self._model is None:
            self._model = pd.read_csv(self._model_path)
        # adding predictions as a column
        # self._preds = self._model[f'preds_from_{self._name}'] = classify_article(, self._model) TODO
        

        
    
        
        
        