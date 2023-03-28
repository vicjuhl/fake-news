import pathlib as pl
import argparse as ap
import pandas as pd
from time import time, localtime, strftime
import json

from model_specific_processing.obj_simple_model import SimpleModel # type: ignore
from model_specific_processing.obj_linear_model import LinearModel # type: ignore
from model_specific_processing.obj_pa_classifier import PaClassifier # type: ignore
from model_specific_processing.obj_meta_model import MetaModel # type: ignore

from model_specific_processing.obj_naive_bayes_models import MultinomialNaiveBayesModel, ComplementNaiveBayesModel  # type: ignore
from model_specific_processing.obj_svm_model import svmModel # type: ignore
from model_specific_processing.obj_random_forest_model import RandomForestModel # type: ignore
from imports.json_to_pandas import json_to_pd # type: ignore
from imports.data_importer import import_val_set, get_split # type: ignore



MODELS: dict = {
    'simple': SimpleModel,
    'linear': LinearModel,
    'pa': PaClassifier,
    'multi_nb': MultinomialNaiveBayesModel,    
    'compl_nb': ComplementNaiveBayesModel,
    'svm': svmModel,
    'random_forest': RandomForestModel,
    'meta_model': MetaModel
}

TRAININGSETS = {
    'simple': 'bow_simple',
    'linear': 'bow_articles',
    'multi_nb': 'bow_articles',    
    'compl_nb': 'bow_articles',
    'pa':'bow_articles',
    'meta_model': 'bow_articles',
    'svm' : 'bow_articles',
    'random_forest': 'bow_articles'
}

METHODNAMES = [
    'train',
    'dump_model',
    'infer4_mm_training',
    'infer',
    'evaluate',
]

def init_argparse() -> ap.ArgumentParser:
    """Initialize the argument parser."""
    parser = ap.ArgumentParser(description='Run a model')
    parser.add_argument('-md', '--models', nargs="*", choices=MODELS.keys(), type=str, default=[], help='Specify list of models')
    # parser.add_argument('--datasets', choices=DATASETS.keys(), help='Dataset to use')
    parser.add_argument('-mt', '--methods', nargs="*", choices=METHODNAMES, default=[], help='Method to run')
    parser.add_argument("-t1", "--train_set_1", nargs="*", help="Splits to include in training set 1")
    parser.add_argument("-t2", "--train_set_2", nargs="*", help="Splits to include in training set 2")
    parser.add_argument("-v", "--val_set", type=int, help="Choose validation set split number")
    parser.add_argument("--test_fake_news", type=str, help="Test models on test data from corpus split 1")
    parser.add_argument("--test_liar", type=str, help="Test models on test data from the LIAR set")
    parser.add_argument("-nt", "--n_train", type=int, default=1000)
    parser.add_argument("-nv", "--n_val", type=int , default=1000)
    parser.add_argument("-hp", "--hyper_params", type=str , default=json.dumps({}))
    return parser

if __name__ == '__main__':
    # Time
    t0_total = time()
    t_session = strftime('%Y-%m-%d_%H-%M-%S', localtime(t0_total))
    # Parsing
    parser = init_argparse()
    args = parser.parse_args()
    # Hyperparameters
    all_params = json.loads(args.hyper_params)
    shared_params = {"n_train": args.n_train, "n_val": args.n_val}
    # Paths
    data_path = pl.Path(__file__).parent.parent.resolve() / "data_files/"
    model_path = pl.Path(__file__).parent.parent.resolve() / "model_files/"
    # Training data
    data_kinds = set([TRAININGSETS[model] for model in args.models])
    training_sets: dict[str, pd.DataFrame] = {}
    # Assert that all split nums are chosen, except 1
    tr1 = [int(num) for num in args.train_set_1]
    tr2 = [int(num) for num in args.train_set_2]
    all_splits = tr1 + tr2 + [args.val_set]
    all_splits.sort()

    if not all_splits == [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        raise ValueError("Some numbers missing in split definitions.")
    
    if "train" in args.methods:
        if "bow_simple" in data_kinds:
            training_sets["bow_simple"] = json_to_pd(args.val_set, 'stop_words_removed')
        if "bow_articles" in data_kinds:
            training_sets["bow_articles"] = pd.read_csv(
                data_path / f"processed_csv/summarized_corpus_valset{args.val_set}.csv",
                nrows=args.n_train
            )
            # Add trn split column based on user input
            bow_art_trn = training_sets["bow_articles"]
            bow_art_trn["trn_split"] = bow_art_trn["split"].apply(
                lambda x: 1 if x in tr1 else 2 if x in tr2 else None
            )
    
    if "infer" in args.methods:
        val_data = import_val_set(
            data_path / 'corpus/reduced_corpus.csv',
            args.val_set,
            get_split(data_path), 
            n_rows = args.n_val # number of rows
        )
    
    for model_name in args.models:
        t0_model = time()
        model_class = MODELS[model_name]
        print("\n", model_class.__name__)
        # Use shared params
        params = shared_params.copy()
        # Add model specialized params
        all_params[model_name] = all_params.get(model_name, {})
        for key, val in all_params[model_name].items():
            params[key] = val
        # Instantiate model
        model_inst = model_class(params, training_sets, args.val_set, model_path, t_session)
        # Run methods
        METHODS = {
            'train': model_inst.train,
            'dump_model': model_inst.dump_model,
            'infer4_mm_training': model_inst.infer4_mm_training,
            'infer': model_inst.infer,
            'evaluate': model_inst.evaluate
        }
        for method_name in args.methods:
            t0 = time()
            print(f"\nRunning method", method_name)  
            
            if method_name == "infer" :
                if isinstance(model_inst, MetaModel):
                    mm_df = pd.read_csv(model_path / 'meta_model/metamodel_train.csv')
                    # REMEMBER this is not the real dataset, SHOULD BE CHANGED!!
                    METHODS[method_name](mm_df)
                else:
                    METHODS[method_name](val_data)
            else:
                METHODS[method_name]()
            print("Runtime", time() - t0)        
        del model_inst
        print("Total model runtime", time() - t0_model)
    print("Total runtime", time() - t0_total)