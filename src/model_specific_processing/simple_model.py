import os
import pandas as pd
from cleantext import clean # type: ignore
import numpy as np
import math 
from utils.functions import stem 
from collections import Counter


def frequency_adjustment(df:pd.DataFrame, total_num_articles) -> pd.DataFrame:
    '''adjusts wordfrequency of all words depending on their labeled'''
    word_freq = df["freq"].apply(lambda x: x[1])
    total_words = word_freq.sum()
    for col in df.columns[1:]: # skip first collumn (contains total frequency)
        local_words = df[col].apply(lambda x: x[1]).sum()
        local_articles = df[col].apply(lambda x: x[0]).sum()
        word_ratio = total_words / local_words # ratio multipled on each word under current label.
        article_ratio = total_num_articles / local_articles
        df[col] = df[col].apply(lambda x: (x[0]*article_ratio,x[1]*word_ratio)) #apply adjustment to all words/article collumns
    return df


def td_idf(df:pd.DataFrame, total_num_articles: int) -> pd.DataFrame:
    '''Total document frequency estimation'''
    Article_freq = df["freq"].apply(lambda x: x[0])
    word_freq = df["freq"].apply(lambda x: 1 if x is None else x[1] if len(x) > 0 else 0)
    
    #df['idf_weight'] = 0
    #To do expects: a dataframe with column "article_frequency"
    for i, row in df.iterrows():
        df.loc[i,"idf_weight"] = np.log(total_num_articles/((row['freq'])[0] + 1.1))
    return df #returns dataframe with weight collumn added.

def logistic_Classification_weight(df:pd.DataFrame) -> pd.DataFrame:
    '''makes a real/fake classifcation between -1 (fake) and 1 (real), 0 being a neutral word'''
    for i, rows in df.iterrows():   
        f = (rows['fake'])[1]
        r = (rows['reliable'])[1]
        x = (r - f) / max(min(r, f), 1) #divided by min of real and fake count but must be atleast 1
        df.loc[i, 'fakeness_score'] = 2 / (1 + math.exp(-x)) - 1
    return df

def Create_model(df: pd.DataFrame) -> pd.DataFrame:
    '''Construct model with weights and scores'''
    #makes new dataframe
    print(df)
    new_df= pd.DataFrame()
    new_df["idf_weight"] = df["idf_weight"]
    new_df["fakeness_score"] = df["fakeness_score"]
    return new_df

def save_to_csv(df: pd.DataFrame):
    """Saves the model to csvfile in Models-csv folder"""
    dir = "models-csv"
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    model_amount = len(os.listdir(dir)) #used for model naming
    Outputfilepath = os.path.join(dir, "Model{}.csv".format(model_amount+1))
    Outputfile = open (Outputfilepath, "w+")
    df.to_csv(Outputfile)
    print("Model saved to csv as: " + "Model" + str(model_amount+1))
    

def build_model(df: pd.DataFrame, article_count: int, make_csv: bool) -> pd.DataFrame:
    """Uses the functions in the module to build a dataframe with weights and scores for words"""
    freq_adj = frequency_adjustment(df, article_count)
    idf = td_idf(freq_adj, article_count)
    log_class = logistic_Classification_weight(idf)
    final_model = Create_model(log_class)
    if make_csv: 
        save_to_csv(log_class)
    return final_model



# inference functions

def binary_classifier(inp: list[(str,int)], df: pd.DataFrame) -> str:

    if any(len(item) != 2 for item in inp):
        raise ValueError("Input data should be a list of tuples with two elements each")
    acc_weight = 0
    acc_score = 0
    for word, freq in inp:
        if word in df.index:
            row = df.loc[str(word)]
            acc_weight += freq * row['idf_weight']
            acc_score +=  row['fakeness_score'] * freq * row['idf_weight']

    return 'fake' if (acc_score / acc_weight) < 0 else 'reliable'



def preproccess_for_inference(article : str) -> list[(str,int)]:
    words = article.split()
    words = [stem(word) for word in words]
    counts = Counter(words)
    counts = [(word, count) for word, count in counts.items()]

    #print(counts)
    return Counter(counts)
    


def infer(input_df: pd.DataFrame, model_df: pd.DataFrame): 
    
    def classifyArticle(inp: str):
        words = preproccess_for_inference(inp)
        return binary_classifier(words, model_df )
    
    input_df["prediction"] = input_df["content"].apply(classifyArticle)
    lst = []
    for index, row in input_df.iterrows():
        if row['type'] == 'fake' or row['type'] == 'reliable':
            lst.append((row['type'], row['prediction'], row['type'] == row['prediction'] ))
    acc = 0

    print(lst)
    for i, j, z in lst:
        if z:
            acc += 1
    accuracy = acc/len(lst)
    print(accuracy)

    return lst, accuracy




    
    