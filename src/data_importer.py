import pathlib as pl
import csv
import sys
from multiprocessing import Pool, cpu_count

from preprocessing.noise_removal import clean_str # type: ignore
from utils.types import news_info, words_info # type: ignore
from preprocessing.words_dicts import WordsDicts # type: ignore

def create_clear_buffer(n_procs: int) -> list[list[news_info]]:
    """Create buffer list with n_procs empty lists."""
    buffer: list[list[news_info]] = []
    for _ in range(n_procs):
        buffer.append([])
    return buffer

def process_batch(articles: list[news_info]) -> list[words_info]:
    """Clean text and split into list of type/bag of words pairs."""
    return [(t, clean_str(c).split(" ")) for t, c in articles]

def process_buffer(buffer: list[list[news_info]], n_procs: int) -> list[words_info]:
    """Multiprocess article buffer, return list of type/bag of words pairs."""
    with Pool(n_procs) as p:
        data_results = p.map_async(process_batch, buffer)
        data_lists = data_results.get()
        # Concattenate list of lists of words to just list of word_info
        return [article for batch in data_lists for article in batch]

def raw_to_words(
    from_file: pl.Path,
    to_path: pl.Path,
    n_rows: int,
    incl_name: str,
    excl_name: str
) -> tuple[int, int, int]:
    """Read raw csv file line by line, clean words, count occurrences and dump to json.
    
    Return tuple of n_included, n_excluded, n_skipped
    """

    csv.field_size_limit(sys.maxsize) # Allow long content texts
    out_dicts = WordsDicts(to_path, incl_name, excl_name)

    n_read: int = 0 # Count rows that were be read.
    n_skipped: int = 0 # Count skipped rows that could not be read.
    n_rows -= 1 # Compensate for 0-indexing

    n_procs = cpu_count()
    batch_sz = 10000
    buffer_sz = n_procs * batch_sz
    # n empty lists
    buffer = create_clear_buffer(n_procs)

    running = True
    i = 0
    
    with open(from_file) as f:
        reader = csv.reader(f)
        next(reader) # skip header
        row = next(reader)

        while running:
            try:
                row = next(reader)
                if i % 5000 == 0:
                    print("Processed lines:", i, "...") # "progress bar"

                # Parallel process data if all batches are full
                if n_read % buffer_sz == 0:
                    out_dicts.add_words(process_buffer(buffer, n_procs))
                    buffer = create_clear_buffer(n_procs)

                # Read and save line
                try:
                    type_ = row[3]
                    content = row[5]
                    # Add article to appropriate batch
                    buffer_index = (n_read % buffer_sz)//batch_sz
                    batch = buffer[buffer_index]
                    batch.append((type_, content))
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
        out_dicts.add_words(process_buffer(buffer, n_procs))

    # Export as json
    out_dicts.export_json()

    return out_dicts.n_incl, out_dicts.n_excl, n_skipped