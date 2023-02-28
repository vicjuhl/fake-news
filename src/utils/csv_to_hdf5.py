# import numpy as np
import pandas as pd
from IPython.display import clear_output
import pathlib as pl

#from ..data_importer import dtypes
dtypes = {
    "id": int,
    "domain": str,
    "type": str,
    "url": str,
    "content": str,
    "scraped_at": str,
    "inserted_at": str,
    "updated_at": str,
    "title": str,
    "authors": str,
    "keywords": str,
    "meta_keywords": str,
    "meta_description": str,
    "tags": str,
    "summary": str
}

CHUNK_SIZE = 1 # default = 5000000 

filename = pl.Path(__file__).parent.parent.parent.resolve() / "data_files/news_sample.csv"

iter_csv = pd.read_csv(
    filename, iterator=True,
    dtype=dtypes, encoding='utf-8', chunksize=CHUNK_SIZE)

cnt = 0
for chunk in iter_csv:
    chunk.to_hdf(
        "data.hdf", 'data', format='table', append=True) #, min_itemsize={'content': 42000}
    cnt += CHUNK_SIZE
    clear_output(wait=True)
    print(f"Processed {cnt:,.0f} coordinates..")