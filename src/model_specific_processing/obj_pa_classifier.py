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
    def __init__(self, training_sets: dict, val_set: int, model_path: pl.Path) -> None:
        super().__init__(training_sets, val_set, model_path)
        self._model = PassiveAggressiveClassifier(max_iter=1000, n_jobs=-1)
        linear_model_path = model_path / "pa_classifier/"
        self._name = "pa_classifier"  
    