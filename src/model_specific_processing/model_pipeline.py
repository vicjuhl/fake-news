import pathlib as pl
import argparse as ap
import pandas as pd
import time
from base_model import BaseModel as bm
from obj_simple_model import SimpleModel
from imports.json_to_pandas import json_to_pd
# run model pipeline

import os
import pickle
from typing import Dict

MODELS: dict = {
    'simple': SimpleModel
}

TRAININGSETS = {
    'simple': 'bag_of_words',
}

METHODS = {
    'data_prep': bm.data_prep,
    'train': bm.train,
    'dump_model': bm.dump_model,
    'infer': bm.infer,
    'evaluate': bm.evaluate,
}

def init_argparse() -> ap.ArgumentParser:
    """Initialize the argument parser."""
    parser = ap.ArgumentParser(description='Run a model')
    parser.add_argument('-m', '--models', choices=MODELS.keys(), help='')
    # parser.add_argument('--datasets', choices=DATASETS.keys(), help='Dataset to use')
    parser.add_argument('-md', '--methods', choices=METHODS.keys(), help='Method to run')
    parser.add_argument("-v", "--val_set", type=int)
    return parser

if __name__ == '__main__':
    # Initialize the argument parser
    parser = init_argparse()
    args = parser.parse_args()   
    
    if args.models not in MODELS:
        raise ValueError(f'Model {args.model} not found')
    else:
        model_classes = [MODELS[model] for model in args.models]
    
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
    
    methods = [METHODS[method] for method in args.method]
    
    for model in model_classes:
        model_inst = model(data_sets, args.val_set)
        for method in methods:
            t0 = time.time()
            print("Running method", method.__name__, "for model", model.__name__)
            model.method()
            print("Runtime", time.time() - t0)
            


    
  
    