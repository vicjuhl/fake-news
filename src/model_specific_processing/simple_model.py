import os
import pandas as pd
from cleantext import clean # type: ignore
import numpy as np
import math 


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
