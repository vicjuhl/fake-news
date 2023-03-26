import pandas as pd
import pickle
import pathlib as pl
import ast
from sklearn.feature_extraction import DictVectorizer # type: ignore
from model_specific_processing.base_model import BaseModel # type: ignore
from preprocessing.noise_removal import preprocess_string # type: ignore
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel(BaseModel):
    '''Random Forest Classification Model'''
    def __init__(
        self,
        params: dict,
        training_sets: dict,
        val_set: int,
        models_dir: pl.Path,
        t_session: str,
    ) -> None:
        super().__init__(params, training_sets, val_set, models_dir, t_session, "random_f", "pkl") # had to choose BaseModel inheritance (instead of LinearModel), since we wish to include the last two parameters here
        self._model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, min_samples_split=5, max_depth=15)
        self._vectorizer = DictVectorizer()
    
    def train(self) -> None:
        '''Trains model on the training data'''
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