import pathlib as pl
import argparse as ap
import pandas as pd
from time import time, localtime, strftime
import json
import ast

from preprocessing.noise_removal import preprocess_string # type: ignore
from model_specific_processing.obj_simple_model import SimpleModel # type: ignore
from model_specific_processing.obj_linear_model import LinearModel # type: ignore
from model_specific_processing.obj_pa_classifier import PaClassifier # type: ignore
from model_specific_processing.obj_meta_model import MetaModel # type: ignore
from model_specific_processing.obj_naive_bayes_models import MultinomialNaiveBayesModel, ComplementNaiveBayesModel  # type: ignore
from model_specific_processing.obj_svm_model import svmModel # type: ignore
from model_specific_processing.obj_random_forest_model import RandomForestModel # type: ignore
from imports.json_to_pandas import json_to_pd # type: ignore
from imports.data_importer import preprocess_val_set, get_split # type: ignore

MODELS: dict = {
    'simple': SimpleModel,
    'linear': LinearModel,
    'pa': PaClassifier,
    'multi_nb': MultinomialNaiveBayesModel,    
    'compl_nb': ComplementNaiveBayesModel,
    'svm': svmModel,
    'random_forest': RandomForestModel,
    'meta': MetaModel
}

TRAININGSETS = {
    'simple': 'bow_simple',
    'linear': 'bow_articles',
    'pa':'bow_articles',
    'multi_nb': 'bow_articles',    
    'compl_nb': 'bow_articles',
    'svm' : 'bow_articles',
    'random_forest': 'bow_articles',
    'meta': 'bow_articles',
}

METHODNAMES = [
    'train',
    'dump_model',
    'infer4_mm_training',
    'infer',
    'evaluate',
]

PREPNAMES = {
    "prep_val"
}

def init_argparse() -> ap.ArgumentParser:
    """Initialize the argument parser."""
    parser = ap.ArgumentParser(description='Run a model')
    parser.add_argument('-md', '--models', nargs="*", choices=MODELS.keys(), type=str, default=[], help='Specify list of models')
    # parser.add_argument('--datasets', choices=DATASETS.keys(), help='Dataset to use')
    parser.add_argument('-mt', '--methods', nargs="*", choices=METHODNAMES, default=[], help='Method to run')
    parser.add_argument("-t1", "--train_set_1", nargs="*", help="Splits to include in training set 1")
    parser.add_argument("-t2", "--train_set_2", nargs="*", help="Splits to include in training set 2")
    parser.add_argument("-p", "--pre_processing", nargs="*", choices=PREPNAMES, help="Preprocess validation data")
    parser.add_argument("-v", "--val_set", type=int, help="Choose validation set split number")
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
    val_data_path = data_path / f"processed_csv/val_data_set{args.val_set}.csv"
    # Training data
    data_kinds = set([TRAININGSETS[model] for model in args.models])
    training_sets: dict[str, pd.DataFrame] = {}
    # Assert that all split nums are chosen, except 1
    tr1 = [int(num) for num in args.train_set_1]
    tr2 = [int(num) for num in args.train_set_2]
    all_splits = tr1 + tr2 + [args.val_set]
    all_splits.sort()

    if not all_splits == [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if args.val_set == 1:
            print("CAUTION: Running on test set!!!")
        elif args.val_set == -1: # LIAR
            print("CAUTION: Running on LIAR set!!!")
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
            bow_art_trn['words'] = bow_art_trn['words'].apply(ast.literal_eval)
            n_fakes = len(bow_art_trn[bow_art_trn["type"] == "fake"])
            n_reals = len(bow_art_trn[bow_art_trn["type"] == "reliable"])
            print(f"Number of fake articles: {n_fakes}, number of reliable articles: {n_reals}")
    
    if "prep_val" in args.pre_processing:
        print("Importing validation data for preprocessing...")
        preprocess_val_set(
            data_path / 'corpus/reduced_corpus.csv',
            val_data_path,
            args.val_set,
            get_split(data_path), 
            n_rows = args.n_val # number of rows
        )
        print("Processed validation data")

    if "infer" in args.methods:
        print("Importing validation data for inference...")
        val_data = pd.read_csv(val_data_path, nrows=args.n_val)
    
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
                    mm_df = pd.read_csv(model_path / 'meta_model/metamodel_inference.csv')
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