import sys
import pandas as pd
from nltk import PorterStemmer # type: ignore

sys.path.append('preprocessing') 
from preprocessing import noise_removal as nr
from preprocessing import words_dicts as wd


# import model 

ps = PorterStemmer()
df = pd.DataFrame

inp = ["hello"]

#stemming function, returns a list of stemmed words

def binary_classifier(inp: list[(str, int)]):
    acc_weight = 0
    acc_score = 0
    for word, freq in inp:
        row = df.loc[str(word)] if word in df.index else None
        if row is not None:
            acc_weight += freq * row['idf_weight']
            acc_score +=  row['fakeness_score'] * freq * row['idf_weight']
        
    return 'Fake' if (acc_score / acc_weight) < 0 else 'True'
            
            
        
        

    
    
    
    
    



