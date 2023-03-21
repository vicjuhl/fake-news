import pandas as pd
import pickle
import pathlib as pl
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression, SGDClassifier #Lasso, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from model_specific_processing.base_model import BaseModel
from imports.data_importer import import_val_set, get_split

class linear_model(BaseModel):
    '''PassiveAggressiveClassifier model'''
    def __init__(self, training_sets: dict, val_set: int, model_path: pl.Path) -> None: # potentially add vectorizer, linear_model as inp
        super().__init__(training_sets , val_set)
        self._model = PassiveAggressiveClassifier(max_iter=1000)
        self._vectorizer = CountVectorizer(stop_words='english')
        self._training_sets = training_sets
        self._name = "linear_model1"
        linear_model_path = model_path / "linear_model/"
        self._data_path =  pl.Path(__file__).parent.parent.resolve() / "data_files/"
        linear_model_path.mkdir(parents=True, exist_ok=True) # Create dest folder if it does not exist
        self._model_path = linear_model_path / f"{self._name}_valset{self._val_set}.pkl"
        self._preds : Optional[pd.DataFrame] = None
        
      
    def train(self) -> None:
        '''Trains a PassiveAggressiveClassifier model on the training data'''
        train_data = self._training_sets["articles"]
        try:
            x_train = train_data['content']
        except KeyError:
            x_train = train_data['shortened']
            
        x_test = train_data['type']
        x_train_vec = self._vectorizer.fit_transform(x_train)
        self._model.fit(x_train_vec, x_test)
             
    def dump_model(self) -> None:
        '''Dumps the model to a pickle file'''
        with open(self._model_path, 'wb') as f:
            pickle.dump(self._model , f)
        print(f'model dumped to {self._model_path}')
   
    def infer(self, df: pd.DataFrame) -> None:
        '''Makes predictions on a dataframe'''
        try:
            with open(self._model_path, 'rb') as f:
                model = pickle.load(f) 
            try:
                df[f'preds_from_{self._name}'] = model.predict(self._vectorizer.transform(df['shortened'])) # adding predictions as a column
            except KeyError:
                df[f'preds_from_{self._name}'] = model.predict(self._vectorizer.transform(df['content'])) # adding predictions as a column
            self._preds = df
        except FileNotFoundError:
            print('cannot make inference without a trained model')    

        
        