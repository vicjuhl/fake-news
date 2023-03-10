import os
from cleantext import clean # type: ignore
import numpy as np
import math 
import sys
import pandas as pd
from nltk import PorterStemmer # type: ignore
sys.path.append('preprocessing') 
from preprocessing import noise_removal as nr
from preprocessing import words_dicts as wd


def frequency_adjustment(df:pd.DataFrame, total_num_articles):
    '''adjusts wordfrequency of all words depending on their labeled'''
    word_freq = df["freq"].apply(lambda x: x[1])
    total_words = word_freq.sum()
    for col in df.columns[1:]: # skip first collumn (contains total frequency)
        local_words = df[col][1].sum()
        local_articles = df[col][0].sum()
        word_ratio = total_words / local_words # ratio multipled on each word under current label.
        article_ratio = total_num_articles / local_articles
        df[col] = df[col].apply(lambda x: (x[0]article_ratio,x[1]word_ratio)) #apply adjustment to all words/article collumns


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


def build(df: pd.DataFrame, article_count: int, save_to_csv: bool) -> pd.DataFrame:
        freq_adj = frequency_adjustment(df, article_count)
        idf = td_idf(freq_adj, article_count)
        log_class = (idf)
        final_model =build_model(log_class, save_to_csv)
        return final_model

    


ps = PorterStemmer()
df = pd.DataFrame

inp = ["hello"]

#stemming function, returns a list of stemmed words

def binary_classifier_simple(inp: list[(str, int)]):
    acc_weight = 0
    acc_score = 0
    for word, freq in inp:
        row = df.loc[str(word)] if word in df.index else None
        if row is not None:
            acc_weight += freq * row['idf_weight']
            acc_score +=  row['fakeness_score'] * freq * row['idf_weight']

    return 'Fake' if (acc_score / acc_weight) < 0 else 'True'


