import pandas as pd
import pickle
import pathlib as pl
import ast
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.feature_extraction import DictVectorizer # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from model_specific_processing.base_model import BaseModel  # type: ignore
from preprocessing.noise_removal import preprocess_string

class LinearModel(BaseModel):
    '''PassiveAggressiveClassifier model'''
    def __init__(self, training_sets: dict, val_set: int, model_path: pl.Path) -> None: # potentially add vectorizer, linear_model as inp
        super().__init__(training_sets , val_set, "linear_model1")
        self._model = LogisticRegression(max_iter=1000)
        self._vectorizer = DictVectorizer()
        self._test_vectorizer = TfidfVectorizer()
        self._training_sets = training_sets
        linear_model_path = model_path / "linear_model/"
        self._data_path =  pl.Path(__file__).parent.parent.resolve() / "data_files/"
        linear_model_path.mkdir(parents=True, exist_ok=True) # Create dest folder if it does not exist
        self._model_path = linear_model_path / f"{self._name}_valset{self._val_set}.pkl"
        self._preds : Optional[pd.DataFrame] = None
        
      
    def train(self) -> None:
        '''Trains a PassiveAggressiveClassifier model on the training data'''
        train_data = self._training_sets["bow_articles"]
        train_data['words_dict'] = train_data['words'].apply(ast.literal_eval) # converting str dict to dict
        y_train = train_data['type']
        x_train_vec = self._vectorizer.fit_transform(train_data['words_dict'].to_list())
        self._model.fit(x_train_vec, y_train)

    def dump_model(self) -> None:
        '''Dumps the model to a pickle file'''
        with open(self._model_path, 'wb') as f:
            pickle.dump(self._model , f)
        print(f'model dumped to {self._model_path}')
   
    def infer(self, df: pd.DataFrame) -> None:
        '''Makes predictions on a dataframe'''
        try:
            if self._model is None:
                with open(self._model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                model = self._model
            df['bow'] = df['content'].apply(lambda x: preprocess_string(x)) # convertingt str to dict[str, int]
            df[f'preds_from_{self._name}'] = model.predict(
                self._vectorizer.transform(df['bow'])
            ) # adding predictions as a column
            self._preds = df
        except FileNotFoundError:
            print('Cannot make inference without a trained model')    

        
        