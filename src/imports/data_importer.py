import pathlib as pl
import csv
import sys
import _csv # for type annotation
from typing import Union, Any, Iterable
from multiprocessing import Pool, cpu_count

from preprocessing.noise_removal import clean_str # type: ignore
from utils.types import news_info, words_info # type: ignore
from utils.mappings import incl_inds # type: ignore
from preprocessing.data_handlers import WordsDicts, CsvWriter # type: ignore

def create_clear_buffer(n_procs: int) -> list[list[news_info]]:
    """Create buffer list with n_procs empty lists."""
    buffer: list[list[news_info]] = []
    for _ in range(n_procs):
        buffer.append([])
    return buffer

def process_buffer(
    out_obj: Union[CsvWriter, WordsDicts],
    buffer: list[list[news_info]],
    n_procs: int,
) -> list[words_info]: # TYPING TODO
    """Multiprocess article buffer, return list of type/bag of words pairs."""
    with Pool(n_procs) as p:
        data_results = p.map_async(out_obj.process_batch, buffer)
        data_lists = data_results.get()
        # Concattenate list of lists of words to just list of word_info
        return [article for batch in data_lists for article in batch]

def process_lines(
    n_rows: int,
    reader: '_csv._reader',
    out_obj: Union[CsvWriter, WordsDicts],
) -> tuple[int, int, int]:
    """Read raw csv file line by line, clean words, count occurrences and dump to json.
    
    Return tuple of n_included, n_excluded, n_skipped
    """
    csv.field_size_limit(sys.maxsize) # Allow long content texts

    n_procs = cpu_count()
    batch_sz = max(1, min(10000, n_rows//n_procs)) # At least 1, at most 10000
    buffer_sz = n_procs * batch_sz # Buffer of n batches of batch_sz
    buffer = create_clear_buffer(n_procs) # n empty lists

    n_read: int = 0 # Count rows that were be read.
    n_skipped: int = 0 # Count skipped rows that could not be read.
    n_rows -= 1 # Compensate for 0-indexing

    running = True
    i = 0

    while running:
        try:
            row = next(reader)
            # "Progress bar"
            if i % 5000 == 0:
                print("Lines read:", i, "...")
            # Parallel process data if all batches are full
            if n_read % buffer_sz == 0:
                out_obj.write(process_buffer(out_obj, buffer, n_procs))
                buffer = create_clear_buffer(n_procs)
            # Read and save line
            try:
                # Add article to appropriate batch
                buffer_index = (n_read % buffer_sz)//batch_sz
                batch = buffer[buffer_index]
                extracted = out_obj.extract(row)
                if extracted != []:
                    batch.append(extracted)
                n_read += 1
            # Or skip row if either type or content cannot be read
            except:
                n_skipped += 1 # ERROR HERE TODO
                print("batch add fail", i, n_skipped)
            # Break when target rows reached
            if i >= n_rows:
                running = False
            i += 1
        except: # No row to read (or other error)
            running = False
    # Flush what remains in the buffer
    out_obj.write(process_buffer(out_obj, buffer, n_procs))
    # Export as json
    out_obj.finalize()
    return out_obj.n_incl, out_obj.n_excl, n_skipped

def raw_to_words(
    from_file: pl.Path,
    to_path: pl.Path,
    n_rows: int,
    incl_name: str,
    excl_name: str
) -> None:
    """Read raw csv file line by line, clean words, count occurrences and dump to json.
    
    Return tuple of n_included, n_excluded, n_skipped.
    """
    to_path.mkdir(parents=True, exist_ok=True) # Create dest folder if it does not exist
    out_dicts = WordsDicts(to_path, incl_name, excl_name)
    with open(from_file) as ff:
        reader = csv.reader(ff)
        next(reader) # skip header
        n_incl, n_excl, n_skipped = process_lines(n_rows, reader, out_obj=out_dicts)
    print(f"{n_incl + n_excl} rows read, \n {n_incl} were included \n {n_excl} were excluded \n {n_skipped} were skipped \n JSON files were written to {to_path}") # PACK away in classes maybe TODO
    if n_incl + n_excl != n_rows:
        raise ValueError("Count numbers do not match")

def reduce_raw(
    from_file: pl.Path,
    to_path: pl.Path,
    n_rows: int
) -> None:
    """Stream-read corpus, process and stream-write to new file.
    
    Return tuple of n_included, n_excluded, n_skipped.
    """
    to_path.mkdir(parents=True, exist_ok=True) # Create dest folder if it does not exist
    with open(from_file) as ff:
        reader = csv.reader(ff)
        row = next(reader) # Get headers
        with open(to_path / "reduced_corpus.csv", 'w') as tf:
            csv_writer = csv.writer(tf)
            csv_writer.writerow(row[i] for i in incl_inds) # Write headers
            writer = CsvWriter(csv_writer)
            n_incl, n_excl, n_skipped = process_lines(n_rows, reader, out_obj=writer)
    print(f"{n_incl + n_excl} rows read, \n {n_incl} were included \n {n_excl} were excluded \n {n_skipped} were skipped \n Reduced csv data file was written to {to_path}")
    if n_incl + n_excl != n_rows:
        raise ValueError("Count numbers do not match")