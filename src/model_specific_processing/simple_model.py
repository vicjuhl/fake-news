import os
import pandas as pd
from cleantext import clean # type: ignore
import numpy as np
import math 
from preprocessing.noise_removal import preprocess_string 
from collections import Counter
import unicodedata


def frequency_adjustment(df:pd.DataFrame) -> pd.DataFrame:
    '''adjusts wordfrequency of all words depending on their labeled'''
    word_freq = df["freq"].apply(lambda x: x[1])
    total_words = word_freq.sum()
    for col in df.columns[1:]: # skip first collumn (contains total frequency)
        local_words = df[col].apply(lambda x: x[1]).sum()
        word_ratio = total_words / local_words # ratio multipled on each word under current label.
        df[col] = df[col].apply(lambda x: (x[0],x[1]*word_ratio)) #apply adjustment to all words
    print('executing function: frequency_adjustment on wordsut with total article count {total_num_articles}')
    return df


def tf_idf(df:pd.DataFrame, total_num_articles: int) -> pd.DataFrame:
    '''Total document frequency estimation'''
    Article_freq = df["freq"].apply(lambda x: x[0])
    word_freq = df["freq"].apply(lambda x: 1 if x is None else x[1] if len(x) > 0 else 0)

    #To do expects: a dataframe with column "article_frequency"
    for i, row in df.iterrows():
        df.loc[i,"idf_weight"] = np.log(total_num_articles/((row['freq'])[0] + 1.1))
    print('excecuting function: tf_idf, applied term frequency adjustment weights')
    return df #returns dataframe with weight collumn added.


def logistic_Classification_weight(df:pd.DataFrame) -> pd.DataFrame:
    '''makes a real/fake classifcation between -1 (fake) and 1 (real), 0 being a neutral word'''
    for i, rows in df.iterrows():   
        f = (rows['fake'])[1]
        r = (rows['reliable'])[1]
        x = np.clip((r - f) / max(min(r, f), 1),-100,100) #divided by min of real and fake count but must be atleast 1
        df.loc[i, 'fakeness_score'] = 2 / (1 + math.exp(-x)) - 1
    print('executing function : logistic_Classification_weight which estimates the fakeness_score of a word')
    return df

def Create_model(df: pd.DataFrame) -> pd.DataFrame:
    '''Construct model with weights and scores'''
    #makes new dataframe
    new_df= pd.DataFrame()
    new_df["idf_weight"] = df["idf_weight"]
    new_df["fakeness_score"] = df["fakeness_score"]
    print('executing function : Create_model that takes ')
    print(new_df.index[2035])
    return new_df

def save_to_csv(df: pd.DataFrame):
    """Saves the model to csvfile in Models-csv folder"""
    dir = "models-csv"
    if not os.path.exists(dir):
        os.makedirs(dir)

    model_amount = len(os.listdir(dir)) #used for model naming
    Outputfilepath = os.path.join(dir, "Model{}.csv".format(model_amount+1))
    Outputfile = open (Outputfilepath, "w+")
    df.index = [unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode('utf-8') for label in df.index]

    df.to_csv(Outputfile, encoding='utf-8', errors='ignore')
    print("Model saved to csv as: " + "Model" + str(model_amount+1))
    

def build_model(df: pd.DataFrame, article_count: int, make_csv: bool) -> pd.DataFrame:
    """Uses the functions in the module to build a dataframe with weights and scores for words"""
    freq_adj = frequency_adjustment(df)
    idf = tf_idf(freq_adj, article_count)
    log_class = logistic_Classification_weight(idf)
    final_model = Create_model(log_class)
    if make_csv: 
        save_to_csv(log_class)
    return final_model


def binary_classifier(words: dict[str, int], df: pd.DataFrame) -> str:
    """Given a dict of words and their freq, and dataframe for simpel model, it makes a binary prediction"""
    acc_weight = 0.1
    acc_score = 0
    df = df.reset_index().drop_duplicates(subset='index', keep='first').set_index('index') # drops all duplicates

    for word, freq in words.items(): 
        if word in df.index:
            row = df.loc[word]
            acc_weight += freq * row['idf_weight']
            acc_score +=  row['fakeness_score'] * freq * row['idf_weight']
            if not isinstance(acc_score,float):
                raise ValueError
        
    return 'fake' if acc_score / acc_weight < 0 else 'reliable' 


def infer(input_df: pd.DataFrame, model_df: pd.DataFrame): 
    """test and validates simple model"""
    print("execute function: infer")
    def classify_article(inp: str) -> str:
        words = preprocess_string(inp)
        return binary_classifier(words, model_df)
    
    mask = (input_df['type'] == 'fake') | (input_df['type'] == 'reliable')
    results_df = input_df[mask]

    results_df["prediction"] = results_df["content"].apply(classify_article)
    
    results_df['correctness'] = (results_df["type"] == results_df["prediction"])
    results_df = results_df.loc[:, ['type', 'prediction', 'correctness']]
    acc = sum(results_df["correctness"])
   
    accuracy = acc/len(results_df)
    print(results_df)
    print("with accuracy",accuracy)

    return results_df, accuracy


    


    
    