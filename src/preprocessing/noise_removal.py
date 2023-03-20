from nltk import PorterStemmer # type: ignore
import pandas as pd
from cleantext import clean # type: ignore
import utils.functions as f # type: ignore
import re

ps = PorterStemmer()

def adding_total_freq(df: pd.DataFrame) -> pd.DataFrame:
    '''Adds a total frequency collumn to the dataframe'''
    df['freq'] = [(f.add_tuples(x, y)) for x, y in zip(df['fake'], df['reliable'])]
    df = df.reindex(columns=['freq', 'fake', 'reliable'])
    print('executing function: adding_total_freq')
    return df 

def cut_tail_and_head(
    df : pd.DataFrame,
    head_quantile: float,
    tail_quantile: float
) -> pd.DataFrame:
    '''Cut the head and tail of the dataframe,
    where the head is the most frequent words and the tail the least frequent words. '''
    df = df.sort_values(by="freq", ascending=False, key=lambda x: x.apply(lambda y: y[1]))

    word_freq = df["freq"].apply(lambda x: x[1])
    total_words = word_freq.sum()   
    acc_index = 0
    acc_sum = 0
    index_upper = 0
    index_lower = 0

    target_sum_head = head_quantile * total_words
    target_sum_tail = (1 - tail_quantile) * total_words   
    while acc_sum < target_sum_head: # finds index of head quantile   
        acc_sum += word_freq[acc_index]
        acc_index += 1
    upper_bound_count = word_freq[acc_index]
    
    while word_freq[acc_index] == upper_bound_count: # continues until frequency changes
        acc_sum += word_freq[acc_index]
        acc_index += 1
    index_upper = acc_index    
        
    while acc_sum < target_sum_tail : # finds index of tail quantile
        acc_sum += word_freq[acc_index]
        acc_index += 1
    lower_bound_count = word_freq[acc_index] 
    
    while word_freq[acc_index] == lower_bound_count: # continues until frequency changes
        acc_index += 1

    index_lower = acc_index
    cut_df = df[index_upper: index_lower]  # remove tail and head from the dataframe
    
    #stats
    uniquewords = len(word_freq) 
    words_left = len(cut_df["freq"])
    words_removed = uniquewords - words_left 

    print(f"Executing function: cut_tail_and_head with quantiles: {head_quantile} and {tail_quantile}")
    print(f"{(head_quantile+tail_quantile)*100}% of unique words removed (APPROX!).")
    print(f"{(words_removed/uniquewords)*100}% of unique words removed.")
    print("unique words before cleaning: ", uniquewords)
    print("unique words after: ", words_left)
    print("unique words removed: ", words_removed)
    print("\tunique words removed from head: ",index_upper)
    print(f"\tunique words removed from tail: {uniquewords - index_lower}")
    print(f"\tat minimum occurence level: {lower_bound_count}")
    return cut_df

def clean_str(text: str) -> str:
    """Clean text for various anomalies."""
    Cleaned= clean(
        text,
        fix_unicode=True,
        to_ascii=True,
        lower=True,
        no_line_breaks=True,
        normalize_whitespace=True,
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_number="<NUM>",
        replace_with_currency_symbol="<CUR>",
        no_punct=True,
    )
    return re.sub("[^\x00-\x7F]+"," ", Cleaned) #removes all nonascii

def tokenize_str(text: str) -> list[str]:
    """Generate list of tokens form string."""
    return text.split()

def stem(tkn: str) -> str:
    """Stem token."""
    return ps.stem(tkn)

def count_words(tkns: list[str]) -> dict[str, int]:
    """Creat dataframe with tokens as rows and frequency as values."""
    counts: dict[str, int] = dict()
    for tkn in tkns:
        counts[tkn] = counts.get(tkn, 0) + 1
    return counts

def preprocess_string(text: str) -> dict[str, int]:
    """Convert a string to a bag of words."""
    clean_text = clean_str(text) 
    tokens = tokenize_str(clean_text)
    stemmed_words = [stem(word) for word in tokens]
    count_dict = count_words(stemmed_words)
    return count_dict

def preprocess_without_stopwords(text: str, incl_words: set[str]) -> dict[str, int]:
    """Convert a string to a bag of words without specified stopwords."""
    cleaned = clean_str(text) 
    tokenized = tokenize_str(cleaned)
    stemmed = [stem(word) for word in tokenized]
    filtered = [word for word in filter(lambda word: word in incl_words, stemmed)]
    count_dict = count_words(filtered)
    return count_dict

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run the preprocessing pipeline."""
    total_freq = adding_total_freq(df)
    no_head_no_tail =(cut_tail_and_head(total_freq, 0.5, 0.05))
    print("--Preprocessing completed--")
    return no_head_no_tail 
