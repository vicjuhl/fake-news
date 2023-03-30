
import pathlib as pl
from model_specific_processing.obj_linear_model import LinearModel # type: ignore
from sklearn.svm import LinearSVC # type: ignore
class svmModel(LinearModel):
    '''Support Vector Classification'''
    def __init__(
        self,
        params: dict,
        training_sets: dict,
        val_set: int,
        models_dir: pl.Path,
        t_session: str,
        name : str = "svm",
        model_format : str = "pkl"
    ) -> None:
        super().__init__(params, training_sets, val_set, models_dir, t_session, name , model_format) # had to choose
        self._model = LinearSVC()
        self._predictor = self._model._predict_proba_lr

    def set_model(self, model) -> None:
        self._model = model
        self._predictor = self._model._predict_proba_lr
