import pandas as pd
from cleantext import clean # type: ignore


def cut_tail_and_head(
    df : pd.DataFrame,
    min_occurence: int,
    head_quantile: float,
    tail_quantile: float
) -> pd.DataFrame:
    '''Cut the head and tail of the dataframe,
    where the head is the most frequent words and the tail the least frequent words. '''

    total_words = df["freq"].sum()   
    acc_index = 0
    acc_sum = 0
    index_upper = 0 
    index_lower = 0 
    
    target_sum_head = head_quantile * total_words
    while acc_sum < target_sum_head: # finds index of head quantile   
        acc_sum += df["freq"][acc_index]
        acc_index += 1
    
    upper_bound_count = df["freq"][acc_index]
    
    while df["freq"][acc_index] == upper_bound_count: # continues until frequency changes
        acc_index += 1
    
    index_upper = acc_index   
    target_sum_tail = (1-tail_quantile) * total_words    
        
    while acc_sum < target_sum_tail and df["freq"][acc_index] > min_occurence: # finds index of tail quantile
        acc_sum += df["freq"][acc_index]
        acc_index += 1
    lower_bound_count = df["freq"][acc_index] 
    
    while df["freq"][acc_index] == lower_bound_count: # continues until frequency changes
        acc_index += 1

    index_lower = acc_index
    cut_df = df[index_upper: index_lower]  # remove tail and head from the dataframe
    
    #stats
    uniquewords = len(df["freq"]) 
    words_left = len(cut_df["freq"])
    words_removed = uniquewords - words_left 

    print("Head and tail cutoff.", "with quantiles: ", 
          head_quantile, " and ", tail_quantile, "i.e", 
          str((head_quantile+tail_quantile)*100)
          + "%" + " of total wordcount removed"
    )
    print("unique words before cleaning: ", uniquewords,  "unique words after: ",
          words_left , "unique words removed: " , words_removed
    )
    print("unique words removed from head: ",index_upper, 
          " unique words removed from tail: ", uniquewords - index_lower,
          "at minimum occurence level: ",lower_bound_count
    )
    return cut
  
def frequency_adjustment(df:pd.DataFrame):
    total = df["freq"].sum()
    for col in df.columns[1:]:
        local = df[col].sum()
        ratio = total/local
        print(ratio)
        df[col] = df[col].apply(lambda x: x*ratio)


def td_idf(df:pd.DataFrame, total_num_articles: int):
    '''total document frequency estimation'''
    df['td_idf_weigh'] = 0
    #To do expects: a dataframe with column "article_frequency"

    for i in range(len(df)):
        df['td_idf_weigh'][i] = np.log(total_num_articles/df['freq_article'][i])*(np.log(df['freq'][i])+1)
    return df


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text for various anomalies for "content" in df."""
    df.content = df.content.apply(lambda x: clean(x,
        lower=True,
        normalize_whitespace=True,
        replace_with_url=True,
        replace_with_email=True,
        replace_with_number=True,
        no_punct=True,
    ))
    return df

def clean_str(text: str) -> str:
    """Clean text for various anomalies."""
    return clean(
        text,
        lower=True,
        normalize_whitespace=True,
        replace_with_url=True,
        replace_with_email=True,
        replace_with_number=True,
        no_punct=True,
    ).replace("\n", "")

def tokenize_str(text: str) -> list[str]:
    """Generate list of tokens form string."""
    return text.split()

def tokenize(df: pd.DataFrame) -> list[str]:
    """Generate list of tokens from dataframe."""
    tkns: list[list[str]] = [tokenize_str(c) for c in df["content"]]
    tkns_combined = [tkn for section in tkns for tkn in section]
    return tkns_combined

def count_sort(tkns: list[str]) -> pd.DataFrame:
    """Creat dataframe with tokens as rows and frequency as values."""
    counts: dict[str, int] = dict()
    for tkn in tkns:
        counts[tkn] = counts.get(tkn, 0) + 1
    df = pd.DataFrame.from_dict(counts, orient="index", columns=["freq"])
    df.sort_values(by=["freq"], ascending=False, inplace=True)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run the preprocessing pipeline."""
    df = clean_text(df)
    tkns = tokenize(df)
    counts = count_sort(tkns)
    no_head_no_tail =(cut_tail_and_head(counts, 10, 0.15, 0.05))
    frequency_adjusted_df = frequency_adjustment(no_head_no_tail) #missing label word count (viktor still works doesnt do anything)
    return frequency_adjusted_df 