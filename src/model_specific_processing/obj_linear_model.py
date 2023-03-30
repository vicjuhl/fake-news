import pandas as pd
import pickle
import pathlib as pl
from sklearn.feature_extraction import DictVectorizer # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from model_specific_processing.base_model import BaseModel # type: ignore
from preprocessing.noise_removal import preprocess_string # type: ignore
from utils.functions import entropy, add_features_df # type: ignore
from sklearn.utils.validation import check_is_fitted # type: ignore
from sklearn.exceptions import NotFittedError # type: ignore
from typing import Any


class LinearModel(BaseModel):
    '''Linear Classifier model'''
    def __init__(
        self,
        params: dict,
        training_sets: dict,
        val_set: int,
        models_dir: pl.Path,
        t_session: str,
        name: str = "linear",
        file_format : str = "pkl"
    ) -> None:
        super().__init__(params, training_sets, val_set, models_dir, t_session, name, file_format)
        self._vectorizer = DictVectorizer()
        try:
            with open(self._savedmodel_path / 'dict_vectorizer.pkl', 'rb') as f:
                self._vectorizer = pickle.load(f)
        except FileNotFoundError as e:
            print("No meta model file found, continuing without vectorizer:", e)
        self._model = LogisticRegression(max_iter=1000, n_jobs=-1)
        self._with_features = True
        self._predictor = self._model.predict_proba
      
    def set_model(self, model: Any) -> None:
        self._model = model
        self._predictor = self._model.predict_proba

    def train(self) -> None:        
        '''Trains a PassiveAggressiveClassifier model on the training data'''
        df: pd.DataFrame = self._training_sets["bow_articles"]
        df = df[df["trn_split"] == 1]
            
        y_train = df['type']
        x_train_vec = self._vectorizer.fit_transform(df['words'].to_list())
        self._model.fit(x_train_vec, y_train)
        
    def dump_model(self) -> None:
        '''Dumps the model to a pickle file'''
        with open(self._model_path, 'wb') as f:
            pickle.dump(self._model , f)
        print(f'model dumped to {self._model_path}')
   
    def infer4_mm_training(self) -> None:
        '''Makes predictions on a dataframe for training of model'''
        try:
            check_is_fitted(self._model)
        except NotFittedError:
            self.load() # loads and sets model
        
        df: pd.DataFrame = self._training_sets["bow_articles"]
        df = df[df["trn_split"] == 2]

        prob_preds = self._predictor(self._vectorizer.transform(df['words'])) #extract probalities
        non_binary_preds = prob_preds[:,1] - prob_preds[:,0] #normalize between 1 (real) and -1 (fake)
        df[f'preds_{self._name}'] = non_binary_preds # adding predictions as a column
                    
        self._preds_mm_training = df[['id', 'type', 'orig_type']].copy()
        # adding predictions as a column
        self._preds_mm_training[f'preds_{self._name}'] = non_binary_preds
        
        #Dumps predictions to a csv for metamodel to train on
        print('generating training data for metamodel, dumping predictions')
        
        self.dump_inference(self._metamodel_train_path, self._preds_mm_training)                
   
    def infer(self, df: pd.DataFrame) -> None:
        '''Makes predictions on a validation dataframe'''
        try:
            check_is_fitted(self._model)
        except NotFittedError:
            self.load() # loads and sets model

        prob_preds = self._predictor(self._vectorizer.transform(df['words']))
        non_binary_preds = prob_preds[:,1] - prob_preds[:,0] #normalize between 1 (real) and -1 (fake)
        self._preds = df[['id', 'type', 'orig_type']].copy()
        self._preds[f'preds_{self._name}'] = non_binary_preds # adding predictions as a column
        
        # adding predictions as a column
        self._preds[f'preds_{self._name}'] = non_binary_preds       
        
        # Dumps the predictions to a csv file
        self.dump_inference(self._metamodel_inference_path, self._preds)
        
    def dump_inference(self, path: pl.Path, preds: pd.DataFrame) -> None:
        '''Dumps the predictions to a csv file'''
        try:
            # load existing metamodel CSV file into a DataFrame
            mm_df = pd.read_csv(path)
        except Exception as e:
            print("Not loading csv: ", e)
            mm_df = pd.DataFrame({'id': preds.id, 'type': preds.type, 'orig_type': preds.orig_type})
            
        try:
            # add new predictions as a new column to existing DataFrame
            col_name = f'preds_{self._name}'
            if col_name not in preds:
                print(f'no predictions to dump for {self._name}')

            if col_name in mm_df.columns:
                mm_df = mm_df.drop(col_name, axis=1) # dropping column if it already exists 
            
            if 'preds_simple_cont' in mm_df.columns:
                mm_df = mm_df.drop('preds_simple_cont', axis=1) # dropping column if it already exists
            
            mm_df = pd.merge(
                mm_df,
                preds.drop(["type", "orig_type"], axis=1), # problem, adding other columns than just preds!!
                on="id",
                how="left",
                suffixes=("_l", "_r")
            )
            # save updated DataFrame to metamodel CSV file
            mm_df.to_csv(path, index=False)
        except Exception as e:
            print(f'Something went wrong adding predictions: {e}')