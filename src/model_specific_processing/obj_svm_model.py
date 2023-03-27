
import pathlib as pl
from sklearn.feature_extraction import DictVectorizer # type: ignore
from model_specific_processing.obj_linear_model import LinearModel # type: ignore
from sklearn.svm import LinearSVC
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
        super().__init__(params, training_sets, val_set, models_dir, t_session, name , model_format) # had to choose BaseModel inheritance (instead of LinearModel), since we wish to include the last two parameters here
        self._model = LinearSVC(probablity=True, n_jobs=-1)
        self._vectorizer = DictVectorizer()
