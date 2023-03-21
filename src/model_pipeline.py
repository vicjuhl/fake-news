import pathlib as pl
import argparse as ap
import pandas as pd
import time
from model_specific_processing.base_model import BaseModel as bm
from model_specific_processing.obj_simple_model import SimpleModel
from model_specific_processing.obj_linear_model import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from imports.json_to_pandas import json_to_pd
from imports.data_importer import import_val_set, get_split

MODELS: dict = {
    'simple': SimpleModel,
    'linear_model': linear_model, 
    #'hashing_vectorizer': linear_model(HashingVectorizer)
}

TRAININGSETS = {
    'simple': 'bow_simple',
    'linear_model': 'bow_articles'
}


def init_argparse() -> ap.ArgumentParser:
    """Initialize the argument parser."""
    parser = ap.ArgumentParser(description='Run a model')
    parser.add_argument('-m', '--models', nargs="*", type=str, help='Specify list of models')
    # parser.add_argument('--datasets', choices=DATASETS.keys(), help='Dataset to use')
    parser.add_argument('-md', '--methods', nargs="*", help='Method to run')
    parser.add_argument("-v", "--val_set", type=int)
    parser.add_argument("-n", "--name", type=int)
    parser.add_argument("-nt", "--n_train", type=int, default= 1000)
    parser.add_argument("-nv", "--n_val", type=int , default= 1000)
    return parser

if __name__ == '__main__':
    # Initialize the argument parser
    t0_total = time.time()
    parser = init_argparse()
    args = parser.parse_args()
    
    model_classes = [MODELS[model] for model in args.models]
    
    data_path = pl.Path(__file__).parent.parent.resolve() / "data_files/"
    model_path = pl.Path(__file__).parent.parent.resolve() / "model_files/"
      
    data_kinds = set([TRAININGSETS[model] for model in args.models])
    training_sets: dict[str, pd.DataFrame] = {}
    if "train" in args.methods:
        if "bow_simple" in data_kinds:
            training_sets["bow_simple"] = json_to_pd(args.val_set)
        if "bow_articles" in data_kinds:
            training_sets["bow_articles"] = pd.read_csv(
                data_path / 'corpus/reduced_corpus.csv',
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
        t0_model = time.time()
        print("\n", model.__name__)
        model_inst = model(training_sets, args.val_set, model_path)

        METHODS = {
            'train': model_inst.train,
            'dump_model': model_inst.dump_model,
            'infer': model_inst.infer,
            'evaluate': model_inst.evaluate,
        }
        for method_name in args.methods:
            t0 = time.time()
            print(f"\nRunning method", method_name)
            if method_name == "infer":
                METHODS[method_name](val_data)
            elif method_name == "evaluate":
                METHODS[method_name]()
                print(val_data.head(2))
            else:       
                METHODS[method_name]()
            print("Runtime", time.time() - t0)        
        del model_inst
        print("Total model runtime", time.time() - t0_model)
    print("Total runtime", time.time() - t0_total)
    
  
    