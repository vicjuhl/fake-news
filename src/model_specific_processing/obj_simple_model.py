import pandas as pd
import pathlib as pl
from typing import Optional
import json

from model_specific_processing.simple_model import frequency_adjustment, tf_idf, logistic_Classification_weight, create_model, classify_article # type: ignore
from model_specific_processing.base_model import BaseModel # type: ignore

class SimpleModel(BaseModel):
    '''Simple model'''
    def __init__(
        self,
        params: dict,
        training_sets: dict,
        val_set: int,
        models_dir: pl.Path,
        t_session: str,
    ) -> None:
        super().__init__(params, training_sets, val_set, models_dir, t_session, "simple")
        self._model: Optional[pd.DataFrame] = None
        
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
        model = self._model
        if model is not None:
            model.to_csv(self._model_path, index=True) 
        else:
            print("ERROR: model could not be dumped")
        print(f"Model saved to {self._model_path}")
        
    def infer(self, test_df) -> None:
        '''Makes predictions on a dataframe'''
        if self._model is None:
            self._model = pd.read_csv(self._model_path, index_col=0) 
        test_df[f'preds_from_{self._name}'] = classify_article(test_df , self._model)         
        self._preds = test_df
        
    
        
        
        