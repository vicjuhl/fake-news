import pathlib as pl
import argparse as ap
import pandas as pd
import time
from model_specific_processing.base_model import BaseModel as bm
from model_specific_processing.obj_simple_model import SimpleModel
from imports.json_to_pandas import json_to_pd

MODELS: dict = {
    'simple': SimpleModel
}

TRAININGSETS = {
    'simple': 'bag_of_words',
}


def init_argparse() -> ap.ArgumentParser:
    """Initialize the argument parser."""
    parser = ap.ArgumentParser(description='Run a model')
    parser.add_argument('-m', '--models', nargs="*", type=str, help='Specify list of models')
    # parser.add_argument('--datasets', choices=DATASETS.keys(), help='Dataset to use')
    parser.add_argument('-md', '--methods', nargs="*", help='Method to run')
    parser.add_argument("-v", "--val_set", type=int)
    return parser

if __name__ == '__main__':
    # Initialize the argument parser
    parser = init_argparse()
    args = parser.parse_args()
    
    model_classes = [MODELS[model] for model in args.models]
    # if args.models not in MODELS:
    #     raise ValueError(f'Model {args.model} not found')
    # else:
    
    data_path = pl.Path(__file__).parent.parent.resolve() / "data_files/"
    # Check if the specified dataset exists
    model_path = data_path = pl.Path(__file__).parent.parent.resolve() / "model_files/"
    # [DATASETS[dataset] for dataset in args.datasets]
    
    data_kinds = set([TRAININGSETS[model] for model in args.models])
    data_sets: dict[str, pd.DataFrame] = {}
    if "train" in args.methods:
        if "bag_of_words" in data_kinds:
            data_sets["bag_of_words"] = json_to_pd(args.val_set)
        # if "articles" in data_kinds:
        #     data_sets["articles"] =
    
    for model in model_classes:
        print("\n", model.__name__)
        model_inst = model(data_sets, args.val_set, model_path)
        METHODS = {
            'train': model_inst.train,
            'dump_model': model_inst.dump_model,
            'infer': model_inst.infer,
            'evaluate': model_inst.evaluate,
        }
        for method_name in args.methods:
            t0 = time.time()
            print(f"\nRunning method", method_name)
            METHODS[method_name]()
            print("Runtime", time.time() - t0)
            


    
  
    