
import pathlib as pl
from sklearn.feature_extraction import DictVectorizer # type: ignore
from model_specific_processing.obj_linear_model import LinearModel # type: ignore
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(LinearModel):
    '''Random Forest Classification Model'''
    def __init__(
        self,
        params: dict,
        training_sets: dict,
        val_set: int,
        models_dir: pl.Path,
        t_session: str,
        name : str = "random_forest",
        model_format : str = "pkl"        
    ) -> None:
        super().__init__(params, training_sets, val_set, models_dir, t_session, name, model_format) # had to choose BaseModel inheritance (instead of LinearModel), since we wish to include the last two parameters here
        self._model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, min_samples_split=3, max_depth=20)
        self._vectorizer = DictVectorizer()
