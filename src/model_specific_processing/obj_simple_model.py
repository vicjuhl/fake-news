import pandas as pd
import pathlib as pl
from typing import Optional
from sklearn.utils.validation import check_is_fitted # type: ignore
from sklearn.exceptions import NotFittedError # type: ignore

from model_specific_processing.simple_model import frequency_adjustment, tf_idf, logistic_Classification_weight, create_model, classify_article_continous # type: ignore
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
        self._name = "simplemodal" 
        super().__init__(params, training_sets, val_set, models_dir, t_session, "simple", "csv")
        self._model: Optional[pd.DataFrame] = None
        
    def set_model(self, model) -> None:
        self._model = model

    def train(self) -> None:
        '''Trains a simple_model instance on the training data'''
        total_num_articles, train_data = self._training_sets["bow_simple"]
        train_data = frequency_adjustment(train_data)
        train_data = tf_idf(train_data, total_num_articles)
        train_data = logistic_Classification_weight(train_data)
        model = create_model(train_data) # creating model dataframe     
        self._model = model
        
    def dump_model(self) -> None:
        '''Dumps the model to a csv file'''
        model = self._model
        saved_path = (self._savedmodel_path / self._name)
        saved_path.mkdir(parents=True, exist_ok=True)
        if model is not None:
            model.to_csv(self._model_path, index=True) 
            model.to_csv(saved_path / ("model" + "." + self.filetype), index=True) 
        else:
            print("ERROR: model could not be dumped")
        print(f"Model saved to {self._model_path}")
        
    def infer(self, df) -> None:
        '''Makes predictions on a dataframe'''
        if self._model is None:
            self.load() # loads and sets model
        
        self._preds = df[['id', 'type', 'orig_type']].copy()
        # adding predictions as a column 
        self._preds[f'preds_{self._name}'] = classify_article_continous(df, self._model)

    def infer4_mm_training(self) -> None:
        """Do nothing."""
        pass
    
        
        