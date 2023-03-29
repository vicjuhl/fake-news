import pathlib as pl
from sklearn.feature_extraction import DictVectorizer # type: ignore
from sklearn.linear_model import PassiveAggressiveClassifier # type: ignore

from model_specific_processing.obj_linear_model import LinearModel # type: ignore


class PaClassifier(LinearModel):
    def __init__(
        self,
        params: dict,
        training_sets: dict,
        val_set: int,
        models_dir: pl.Path,
        t_session: str,
        name : str = "pa_classifier",
        model_format : str = "pkl"
    ) -> None:
        super().__init__(params, training_sets, val_set, models_dir, t_session, name, model_format)
        self._model = PassiveAggressiveClassifier(max_iter=1000, n_jobs=-1)
        self._predictor = self._model._predict_proba_lr
