import pandas as pd
import numpy as np
import math
from preprocessing.noise_removal import preprocess_string # type: ignore

def frequency_adjustment(df:pd.DataFrame) -> pd.DataFrame:
    '''Adjust wordfrequency of all words depending on their label.'''
    word_freq = df["freq"].apply(lambda x: x[1])
    total_words = word_freq.sum()
    for col in df.columns:
        if col is not df["freq"]:
            local_words = df[col].apply(lambda x: x[1]).sum()
            word_ratio = total_words / local_words # ratio multiplied on each words freq, under the current label.
            df[col] = df[col].apply(lambda x: (x[0], x[1] * word_ratio)) #apply adjustment to all words
    print('executing function: frequency_adjustment on wordsut with total article count')
    return df

def tf_idf(df:pd.DataFrame, total_num_articles: int) -> pd.DataFrame:
    '''Total document frequency estimation.'''
    print('excecuting function: tf_idf, applied term frequency adjustment weights')
    for i, row in df.iterrows():
        df.loc[i,"idf_weight"] = np.log(total_num_articles/((row['freq'])[0] + 1.001))
    return df #returns dataframe with weight collumn added.

def logistic_Classification_weight(df:pd.DataFrame) -> pd.DataFrame:
    '''Make a real/fake classifcation between -1 (fake) and 1 (real), 0 being a neutral word.'''
    for i, rows in df.iterrows():   
        f = (rows['fake'])[1]
        r = (rows['reliable'])[1]
        x = np.clip((r - f) / max(min(r, f), 1),-100,100) #divided by min of real and fake count but must be atleast 1
        df.loc[i, 'fakeness_score'] = 2 / (1 + math.exp(-x)) - 1
    return df

def create_model(df: pd.DataFrame) -> pd.DataFrame:
    '''Construct model with weights and scores'''
    new_df= pd.DataFrame()
    new_df["idf_weight"] = df["idf_weight"]
    new_df["fakeness_score"] = df["fakeness_score"]
    return new_df

def binary_classifier(words: dict[str, int], df: pd.DataFrame) -> str:
    """Given a dict of words and their freq, and dataframe for simpel model, it makes a binary prediction."""
    acc_weight = 0.1
    acc_score = 0

    for word, freq in words.items(): 
        if word in df.index:
            row = df.loc[word]
            acc_weight += freq * row['idf_weight']
            acc_score +=  row['fakeness_score'] * freq * row['idf_weight']
            if not isinstance(acc_score,float):
                raise ValueError
    # the following division produces an average (no effect on binary classification)   
    return 'fake' if acc_score / acc_weight < 0 else 'reliable' # shoudl be changed to non binary 

def classify_article(val_df: pd.DataFrame, model_df: pd.DataFrame) -> list[str]:
    """Classifies all articles in the input dataframe, and returns a list of predictions."""
    predictions = []
    column = val_df['content'].apply(lambda x: preprocess_string(x))
    column.apply(lambda x: predictions.append(binary_classifier(x, model_df)))       
    
    return predictions
    
