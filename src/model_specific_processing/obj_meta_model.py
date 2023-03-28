from sklearn.linear_model import LogisticRegression # type: ignore
import pathlib as pl 
from typing import Optional
import pandas as pd
from sklearn.feature_extraction import DictVectorizer # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
import pickle
from utils.functions import create_dict_MetaModel # type: ignore
from model_specific_processing.base_model import BaseModel  # type: ignore


class MetaModel(BaseModel):
    def __init__(
         self,
        params: dict,
        training_sets: dict,
        val_set: int,
        models_dir: pl.Path,
        t_session: str,
        name : str = "meta_model",
        model_format : str = "pkl"
    ) -> None:
        super().__init__(params, training_sets, val_set, models_dir, t_session, name, model_format)
        self._model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0, n_jobs=-1)
        self._vectorizer = DictVectorizer()
        
    def train(self) -> None:
        '''Train the model on the given training sets'''
        try:            
            train_data = pd.read_csv(self._metamodel_train_path)
            
            labels = train_data['type'] # strings
            labels = labels.apply(
                lambda x:
                1 if x == 'reliable' else
                # -1 if x == 'fake' else
                # None TODO: 
                -1
            )
            train_data.drop(['id', 'type'], axis = 1, inplace=True) # should not be used for training because of information polution      

            # put predictions from models as key-value pairs in a dictionary
            train_data['dict'] =  train_data.apply(create_dict_MetaModel, axis=1) 
            train_data_vec = self._vectorizer.fit_transform(train_data['dict'].to_list())            
            self._model.fit(train_data_vec, labels)    

        except FileNotFoundError or pd.errors.EmptyDataError:
            print('metamodel cannot be trained, empty csv')   
             
    def dump_for_mm_training(self):
        pass   
    
    def dump_model(self) -> None:
        '''Dumps the model to a pickle file'''
        with open(self._model_path, 'wb') as f:
            pickle.dump(self._model , f)
        print(f'model dumped to {self._model_path}')
        
    def infer(self, df: pd.DataFrame) -> None:
        '''Infer the model on the given dataframe'''
        # load model
        try:   
            if self._model is None:
                with open(self._model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                model = self._model
            
            self._preds = df        
            labels = df['type'] # saving type          
            df.drop(['id', 'type'], axis = 1, inplace=True)            
            df = df.applymap(lambda x:
                1 if x == 'reliable' else
                0 if x == 'fake' else
                x
            )
            df['inference_column'] = df.apply(create_dict_MetaModel, axis=1) 
            vec = self._vectorizer.fit_transform(df['inference_column'].to_list())            
            self._preds[f'preds_{self._name}'] = model.predict(vec)
            self._preds['type'] = labels # restoring type
        
        except FileNotFoundError:
            print('Cannot make inference without a trained model')    

        