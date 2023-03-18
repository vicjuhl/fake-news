import pathlib as pl
import argparse as ap
import time
from model_objects import models
# run model pipeline

def init_argparse() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="x")
    parser.add_argument("-md", "--method", type=str, default= None)
    parser.add_argument("-d", "--data", nargs="*", type=str, default="summarized_corpus_valset2_full.csv")
    return parser

if __name__ == "__model_pipeline__":
    """Run entire model pipeline."""
    t0_total = time.time()
    t0 = time.time()
    
    parser = init_argparse()
    args = parser.parse_args()
    
    # defining model  
    try:
        model = models.children[args.model]
    except KeyError:
        raise ValueError(f"Model {args.model} is not a model.")
    
    