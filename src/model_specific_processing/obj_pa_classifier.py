import pandas as pd
import pickle
import pathlib as pl
import ast
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.feature_extraction import DictVectorizer # type: ignore
from sklearn.linear_model import PassiveAggressiveClassifier # type: ignore
from model_specific_processing.base_model import BaseModel  # type: ignore
import time 

from model_specific_processing.obj_linear_model import LinearModel 

class PaClassifier(LinearModel):
    def __init__(
        self,
        params: dict,
        training_sets: dict,
        val_set: int,
        models_dir: pl.Path,
        t_session: str,
    ) -> None:
        super().__init__(params, training_sets, val_set, models_dir, t_session, "pa-classifier", "pkl")
        def __init__(
        self,
        params: dict,
        training_sets: dict,
        val_set: int,
        models_dir: pl.Path,
        t_session: str,
    ) -> None:
            super().__init__(params, training_sets, val_set, models_dir, t_session, "pa_classifier", "pkl")
            self._vectorizer = DictVectorizer()
            self._model = PassiveAggressiveClassifier(max_iter=1000, n_jobs=-1)
            self._training_sets = training_sets
            self._name = "pa_classifier"  
    