import pathlib as pl
from sklearn.feature_extraction import DictVectorizer # type: ignore
from sklearn.naive_bayes import MultinomialNB, ComplementNB # type: ignore
from model_specific_processing.obj_linear_model import LinearModel # type: ignore


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
        name :str = "multi_nb",
        file_format : str = "pkl"
    ) -> None:
        super().__init__(params, training_sets, val_set, models_dir, t_session, name , file_format) # had to choose BaseModel inheritance (instead of LinearModel), since we wish to include the last two parameters here
        self._model = MultinomialNB(alpha=1, force_alpha=True)
        self._vectorizer = DictVectorizer()
        self._predictor = self._model.predict_proba
           
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
        super().__init__(params, training_sets, val_set, models_dir, t_session, "compl_nb", "pkl") # had to choose BaseModel inheritance (instead of LinearModel), since we wish to include the last two parameters here
        self._model = ComplementNB(alpha=1, force_alpha=True)
        self._vectorizer = DictVectorizer()
        self._predictor = self._model.predict_proba