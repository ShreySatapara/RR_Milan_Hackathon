from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import csv
from scipy.special import softmax
import urllib.request

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)


# given array of strings, returns avg values [neg_senti, neut_senti, pos_senti]
def get_senti(text):
    final_scores = np.array([0,0,0])
    for t in text:
        encoded_input = tokenizer(t, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        final_scores = np.add(final_scores,scores)
    print(final_scores/len(text))
    return final_scores #NEG NU POS

text = ["go to hell","good morning",'how are you?']             # sample array of texts(strings) 
get_senti(text)
