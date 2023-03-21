import pathlib as pl
import argparse as ap
import pandas as pd
from time import time, localtime, strftime
from model_specific_processing.obj_simple_model import SimpleModel # type: ignore
from model_specific_processing.obj_linear_model import LinearModel # type: ignore
from imports.json_to_pandas import json_to_pd # type: ignore
from imports.data_importer import import_val_set, get_split # type: ignore

MODELS: dict = {
    'simple': SimpleModel,
    'linear': LinearModel, 
    #'hashing_vectorizer': linear_model(HashingVectorizer)
}

TRAININGSETS = {
    'simple': 'bow_simple', # tuple of int, df
    'linear': 'bow_articles'
}

METHODNAMES = [
    'train',
    'dump_model',
    'infer',
    'evaluate',
]

def init_argparse() -> ap.ArgumentParser:
    """Initialize the argument parser."""
    parser = ap.ArgumentParser(description='Run a model')
    parser.add_argument('-md', '--models', nargs="*", choices=MODELS.keys(), type=str, help='Specify list of models')
    # parser.add_argument('--datasets', choices=DATASETS.keys(), help='Dataset to use')
    parser.add_argument('-mt', '--methods', nargs="*", choices=METHODNAMES, help='Method to run')
    parser.add_argument("-v", "--val_set", type=int)
    parser.add_argument("-nt", "--n_train", type=int, default=1000)
    parser.add_argument("-nv", "--n_val", type=int , default=1000)
    return parser

if __name__ == '__main__':
    # Initialize the argument parser
    t0_total = time()
    t_session = strftime('%Y-%m-%d_%H-%M-%S', localtime(t0_total))
    parser = init_argparse()
    args = parser.parse_args()
    shared_params = {"n_train": args.n_train, "n_val": args.n_val}
    
    model_classes = [MODELS[model] for model in args.models]
    
    data_path = pl.Path(__file__).parent.parent.resolve() / "data_files/"
    model_path = pl.Path(__file__).parent.parent.resolve() / "model_files/"
      
    data_kinds = set([TRAININGSETS[model] for model in args.models])
    training_sets: dict[str, pd.DataFrame] = {}
    if "train" in args.methods:
        if "bow_simple" in data_kinds:
            training_sets["bow_simple"] = json_to_pd(args.val_set, 'stop_words_removed')
        if "bow_articles" in data_kinds:
            training_sets["bow_articles"] = pd.read_csv(
                data_path / f"processed_csv/summarized_corpus_valset{args.val_set}.csv",
                nrows=args.n_train
            )
    
    if "infer" in args.methods:
        val_data = import_val_set(
            data_path / 'corpus/reduced_corpus.csv',
            args.val_set,
            get_split(data_path), 
            n_rows = args.n_val # number of rows
        )
    
    for model in model_classes:
        t0_model = time()
        print("\n", model.__name__)
        params = shared_params.copy()
        model_inst = model(params, training_sets, args.val_set, model_path, t_session)

        METHODS = {
            'train': model_inst.train,
            'dump_model': model_inst.dump_model,
            'infer': model_inst.infer,
            'evaluate': model_inst.evaluate,
        }
        for method_name in args.methods:
            t0 = time()
            print(f"\nRunning method", method_name)
            if method_name == "infer":
                METHODS[method_name](val_data)
            else:
                METHODS[method_name]()
            print("Runtime", time() - t0)        
        del model_inst
        print("Total model runtime", time() - t0_model)
    print("Total runtime", time() - t0_total)