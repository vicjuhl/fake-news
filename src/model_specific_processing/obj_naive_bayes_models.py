import pandas as pd
import pickle
import pathlib as pl
import ast
from sklearn.feature_extraction import DictVectorizer # type: ignore
from sklearn.naive_bayes import MultinomialNB, ComplementNB # type: ignore
from model_specific_processing.obj_linear_model import LinearModel # type: ignore
from preprocessing.noise_removal import preprocess_string # type: ignore

class MultinomialNaiveBayesModel(LinearModel):
    '''Multinomial Naive Bayes model
    
    suitable for classification with discrete features (e.g., word counts for text classification).
    '''
    def __init__(
        self,
        params: dict,
        training_sets: dict,
        val_set: int,
        models_dir: pl.Path,
        t_session: str,
    ) -> None:
        super().__init__(params, training_sets, val_set, models_dir, t_session, "multinomial Naive Bayes", "pkl")
        self._model = MultinomialNB(alpha=1, force_alpha=True)
        self._vectorizer = DictVectorizer()


class ComplementNaiveBayesModel(LinearModel):
    '''Complement Naive Bayes model
    
    The Complement Naive Bayes classifier was designed to correct the “severe assumptions” made by the 
    standard Multinomial Naive Bayes classifier. It is particularly suited for imbalanced data sets.
    '''
    def __init__(
        self,
        params: dict,
        training_sets: dict,
        val_set: int,
        models_dir: pl.Path,
        t_session: str,
    ) -> None:
        super().__init__(params, training_sets, val_set, models_dir, t_session, "complement Naive Bayes", "pkl")
        self._model = ComplementNB(alpha=1, force_alpha=True)
        self._vectorizer = DictVectorizer()

