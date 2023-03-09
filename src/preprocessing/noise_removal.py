import pandas as pd
from cleantext import clean # type: ignore
import numpy as np
import math 

def adding_total_freq(df: pd.DataFrame) -> pd.DataFrame:
    '''Adds a total frequency collumn to the dataframe'''
    df['total_freq'][1] = df["fake"].apply(lambda x: x[1]) + df["real"].apply(lambda x: x[1])
    df['total_freq'][0] = df["fake"].apply(lambda x: x[0]) + df["real"].apply(lambda x: x[1])
    return df


def cut_tail_and_head(
    df : pd.DataFrame,
    min_occurence: int,
    head_quantile: float,
    tail_quantile: float
) -> pd.DataFrame:
    '''Cut the head and tail of the dataframe,
    where the head is the most frequent words and the tail the least frequent words. '''

    word_freq = df["freq"].apply(lambda x: x[1])
    total_words = word_freq.sum()   
    acc_index = 0
    acc_sum = 0
    index_upper = 0 
    index_lower = 0 
    
    target_sum_head = head_quantile * total_words
    while acc_sum < target_sum_head: # finds index of head quantile   
        acc_sum += word_freq[acc_index]
        acc_index += 1
    
    upper_bound_count = word_freq[acc_index]
    
    while word_freq[acc_index] == upper_bound_count: # continues until frequency changes
        acc_index += 1
    
    index_upper = acc_index   
    target_sum_tail = (1-tail_quantile) * total_words    
        
    while acc_sum < target_sum_tail and word_freq[acc_index] > min_occurence: # finds index of tail quantile
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
    return cut_df
  
def frequency_adjustment(df:pd.DataFrame, total_num_articles):
    '''adjusts wordfrequency of all words depending on their labeled'''
    word_freq = df["freq"].apply(lambda x: x[1])
    total_words = word_freq.sum()
    for col in df.columns[1:]: # skip first collumn (contains total frequency)
        local_words = df[col][1].sum()
        local_articles = df[col][0].sum()
        word_ratio = total_words / local_words # ratio multipled on each word under current label.
        article_ratio = total_num_articles / local_articles
        df[col] = df[col].apply(lambda x: (x[0]*article_ratio,x[1]*word_ratio)) #apply adjustment to all words/article collumns


def td_idf(df:pd.DataFrame, total_num_articles: int):
    '''Total document frequency estimation'''
    Article_freq = df["freq"].apply(lambda x: x[0])
    word_freq = df["freq"].apply(lambda x: x[1])
    
    df['idf_weight'] = 0
    #To do expects: a dataframe with column "article_frequency"

    for i in range(len(df)):
        df['idf_weight'][i] = np.log(total_num_articles/Article_freq[i])*(np.log(word_freq[i])+1)
    return df #returns dataframe with weight collumn added.

def logistic_Classification_weight(df:pd.DataFrame ):
    '''makes a real/fake classifcation between -1 (fake) and 1 (real), 0 being a neutral word'''
    fake = df["fake-freq"].apply(lambda x: x[1])
    real =df["real-freq"].apply(lambda x: x[1])
    scores = [] 
    for f,r in zip(fake, real): 
        x = (r - f) / min(r, f)
        score= 1/(1+ math.exp(-x))
        scores.append(score)
    df['fakeness_score'] = scores
    return df

import os

def build_model(df: pd.DataFrame, save_to_csv : bool) -> pd.DataFrame:
    '''Construct model with weights and scores, and '''
    #makes new dataframe
    new_df= pd.DataFrame()
    new_df["idf_weight"] = df["idf_weight"]
    new_df["fakeness_score"] = df["fakeness_score"]

    #makes new csv file and outputs the model
    if save_to_csv:
        dir = "models"
        model_amount = len(os.listdir(dir)) #used for model naming
        Outputfilepath = os.path.join(dir, "Model{}.csv".format(model_amount+1))
        Outputfile = open (Outputfilepath, "w+")
        new_df.to_csv(Outputfile)
        print("Model saved to csv as: " + "Model" + str(model_amount+1))
    return new_df

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

def tokenize(df: pd.DataFrame) -> list[str]:
    """Generate list of tokens from dataframe."""
    tkns: list[list[str]] = [c.split(" ") for c in df["content"]]
    tkns_combined = [tkn for section in tkns for tkn in section]
    return tkns_combined

def count_sort(tkns: list[str]) -> pd.DataFrame:
    """Creat dataframe with tokens as rows and frequency as values."""
    counts: dict[str, int] = dict()
    for tkn in tkns:
        counts[tkn] = counts.get(tkn, 0) + 1
    df = pd.DataFrame.from_dict(counts, orient="index", columns=["freq"])
    df.sort_values(by=["freq"][1], ascending=False, inplace=True)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run the preprocessing pipeline."""
    df = clean_text(df)
    tkns = tokenize(df)
    counts = count_sort(tkns)

    no_head_no_tail =(cut_tail_and_head(counts, 10, 0.50, 0.05))
    frequency_adjusted_df = frequency_adjustment(no_head_no_tail) #missing label word count (viktor still works doesnt do anything)
    return frequency_adjusted_df 

preprocess(pd.DataFrame)