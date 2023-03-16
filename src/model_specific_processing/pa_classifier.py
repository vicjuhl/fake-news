
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


# define model if not already existing
try:
    with open('model.pkl', 'rb') as f:
        pac = pickle.load(f)
except FileNotFoundError:
    pac = PassiveAggressiveClassifier(max_iter=50) # PassiveAggressiveClassifier

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) # initialize vectorizer 

# expecting dataframe with content and type columns
def train_pac(data : pd.DataFrame):
    '''trains a PassiveAggressiveClassifier model on the training data'''
    global tfidf_vectorizer

    x_train = data['content']
    x_test = data['type'] # column with labels, OBS could have used!!
    # split data into train and test this might be a bit too much since we have own test_dataset
    
    tfidf_train= tfidf_vectorizer.fit_transform(x_train)  
       
    pac.fit(tfidf_train, x_test) # train model
    with open('model.pkl', 'wb') as f:
        pickle.dump(pac, f)


def infer_pac(df : pd.DataFrame):
    global tfidf_vectorizer
    tfidf = tfidf_vectorizer.transform(df['content']) # preprocess data
    df['pac_preds'] = pac.predict(tfidf) # new column in df with predictions
    return df 
