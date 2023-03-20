### **Word2vec and GloVe Embeddings**

#To run this notebook I use genism=3.8.3, the updated version changes some function, wich won't work for this notebook
%pip install pip install --user gensim==3.8.3


#Import the necesarry libraries

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import random
import seaborn as sns

import re, string, unicodedata                          
import contractions                                     
from bs4 import BeautifulSoup 
from collections import Counter         
from tqdm import tqdm


import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize  
from nltk.stem.wordnet import WordNetLemmatizer         
import matplotlib.pyplot as plt                         
import seaborn as sns
import matplotlib.pyplot as plt

import spacy
from spacy import displacy
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.lang.es.stop_words import STOP_WORDS
import gensim

from gensim.models import Word2Vec
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = "{:,.2f}".format
pd.options.display.precision = 3
pd.set_option('display.max_colwidth', None)

import warnings
warnings.filterwarnings("ignore")

print(print(gensim.__version__))

#Load the dataset
data = pd.read_csv('corpus_mx.csv')

#Check the data
data.head()
data.info()

#Check values in sentiment variable 
data.sentiment.value_counts()

# Merge values for positive, negativa and neutral sentiment
data = data.copy()

data['sentiment'] = data['sentiment'].replace(['N'], 'NEG')
data.sentiment.value_counts()

data['sentiment'] = data['sentiment'].replace(['P'], 'POS')
data.sentiment.value_counts()

data['sentiment'] = data['sentiment'].replace(['NONE'], 'NEU')
data.sentiment.value_counts()

#Let's visualize the labels 
sns.countplot(x=data['sentiment'], data=data, palette='Blues')

# Let's look at the top 150 unique words in the data set

all_texts = " ".join(texts for texts in data.text)
print ("There are {} words in the combination of all texts.".format(len(all_texts)))

# Lower max_font_size, change the maximum number of word and lighten the background:
wordcloud = WordCloud(max_font_size = 40, max_words = 150, background_color="white").generate(all_texts)
plt.figure(figsize = (8,16))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()



#DATA PRE-PROCESSING STEP 1

# We will create multiple functions for this step and then encompass all those functions into a single helper one.

stopwords = list(STOP_WORDS)

#remove the html tags
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")                    
    return soup.get_text()


#remove the numericals present in the text
def remove_numbers(text):
  text = re.sub(r'\d+', '', text)
  return text

# remove the url's present in the text
def remove_url(text): 
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',text)
    return text

# remove the mentions in the text
def remove_mention(text):
    text = re.sub(r'@\w+','',text)
    return text

#remove specific words
def remove_specificwords(text):
           text = re.sub("url", '', text)
           text = re.sub("emoji", '', text)
           text = re.sub("ja", '', text)
           text = re.sub("jaja", '', text)
           text = re.sub("jajaja", '', text)
           text = re.sub("jajajaja", '', text)
           text = re.sub("jajajajaja", '', text)
           text = re.sub("jajajajajaja", '', text)
           text = re.sub("jajajajajajaja", '', text)
           text = re.sub("jajajajajajajaja", '', text)
           text = re.sub("jajajajajajajajaja", '', text)
           text = re.sub("jajajajajajajajajaja", '', text)
           return text

def remove_tilde(text):
    text = re.sub('\\u00f1', 'ñ',text)
    text = re.sub('\\u00e1', 'a',text)
    text = re.sub('\\u00e9', 'e',text)
    text = re.sub('\\u00ed', 'i',text)
    text = re.sub('\\u00f3', 'o',text)
    text = re.sub('\\u00fa', 'u',text)
    text = re.sub('\\u00bf', '¿',text)
    text = re.sub('\\u00a1', '¡',text)
    text = re.sub('\\u00d1', 'Ñ',text)
    text = re.sub('\\u00c1', 'A',text)
    text = re.sub('\\u00c9', 'E',text)
    text = re.sub('\\u00cd', 'I',text)
    text = re.sub('\\u00d3', 'O',text)
    text = re.sub('\\u00da', 'U',text)
    text = re.sub('\\u00fc', 'ü',text)
    text = re.sub('\\u00b0', '',text)    
    return text


def to_lower(text):
    text = text.lower()
    return text

def remove_puntctuation(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text

def remove_non_ascii(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_hash(text):
    text = re.sub(r'#\w+','', text)
    return text

#function that encompasses all of the above
def clean_text(text):
    text = strip_html(text)
    text = remove_numbers(text)
    text = remove_url(text)
    text = remove_mention(text)
    text = remove_specificwords(text)
    text = remove_tilde(text)
    text = to_lower(text)
    text = remove_puntctuation(text)
    text = remove_non_ascii(text)
    text = remove_hash(text)
    return text

data['text'] = data['text'].apply(lambda x: clean_text(x))


#DATA PRE-PROCESSING STEP 2

#Remove particular and specifi words related to the data set

def remove_particularwords(text):
            text = re.sub("url", '', text)
            text = re.sub("emoji cara", '', text)
            text = re.sub("risa emoji", '', text)
            text = re.sub("revolviéndose de la risa emoji", '', text)
            text = re.sub("emoji cara llorando", '', text)
            text = re.sub("ojo", '', text)
            text = re.sub("mano", '', text)
            text = re.sub("manos", '', text)
            text = re.sub("ojos", '', text)
            text = re.sub("ojos sonrientes", '', text)
            text = re.sub("sonriendo", '', text)
            text = re.sub("emoji manos", '', text)
            text = re.sub("emoji tono de piel", '', text)
            text = re.sub("aplaudiendo", '', text)
            text = re.sub("manos abrazando", '', text)
            text = re.sub("manos aplaudiendo", '', text)
            text = re.sub("mano sobre la boca", '', text)
            text = re.sub("cara feliz", '', text)
            text = re.sub("cara sonriendo", '', text)
            text = re.sub("sonriendo", '', text)
            text = re.sub("emoji bandera méxico", '', text)
            text = re.sub("emoji cara con", '', text)
            text = re.sub("emoji cara con mano", '', text)
            text = re.sub("emoji cara con mano sobre la boca", '', text)
            text = re.sub("tono de piel", '', text)
            text = re.sub("girasol", '', text)
            text = re.sub("bandera mexico", '', text)
            text = re.sub("emoji corazón birllante", '', text)
            text = re.sub("cara", '', text)
            text = re.sub("risa", '', text)
            text = re.sub("revolviéndose", '', text)
            text = re.sub("llorando", '', text)
            text = re.sub("boca", '', text)
            text = re.sub("gafas de sol", '', text)
            text = re.sub("revolviendose", '', text)
            text = re.sub("bandera", '', text)
            text = re.sub("mexico", '', text)
            text = re.sub("guiñando", '', text)
            text = re.sub("guinando", '', text)
            text = re.sub("pulgar", '', text)
            text = re.sub("dorso", '', text)
            return text

data['text'] = data['text'].apply(lambda x: remove_particularwords(x))



data.head()

#DATA PRE-PROCESSING STEP 3

# Remove stop words (spanish stop words)
from spacy.lang.es.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)

#Create a final corpus, splitting and appending the processed text into a new variable, this step is crucial, 
#because if we try to remove the stop word separately the function (stop words) will yield separate letters;
#that is why we need to split, join and append the text. 

final_corpus = []

for i in range(data.shape[0]):
    
    # To remove the numbers and other arabic numeric symbols
    text = re.sub('[^a-zA-Z0-9]', ' ', data['text'][i])

    # Splitting the text
    text = text.split()
    
    # Removing the stopwords
    text = [word for word in text if not word in stopwords]
    
    # joining the words into text
    text = ' '.join(text)
    
    # appending the text to the final_corpus list
    final_corpus.append(text)
    
data['final_text'] = final_corpus

data.head()

#Drop the text column 
data = data.drop(['text'], axis = 1)
data = data[['final_text', 'sentiment']]
data.head()


#Word2vec

#First we need to create a list of words and then append it so that this list can serve as a input for the model. 

words_list =[]

for i in data['final_text']:
    
    li = list(i.split(" "))
    
    words_list.append(li)


#Creating word embeddings

#Word2vec takes the following three important parameters:
#Min_count**: It will ignore all the words with a total frequency lower than this.
#Workers**: These are the threads to train the model.

#First we need to pip install version 3.8.3 pf gensim



#Second, create the model and save it
model= Word2Vec(words_list, min_count = 1, workers = 4)
model.save("word2vec.model")

#Number of words in vocabulary
words = model.wv.vocab
len(words)

from gensim.models import Word2Vec

# Helper function to get a document with all the embeddings

def average_word_vectors(words, model, vocabulary, num_features):
    
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.
    
    for word in words:
        if word in vocabulary: 
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model[word])
    
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
        
    return feature_vector
    
   
def averaged_word_vectorizer(corpus, model, num_features):
    
    vocabulary = set(model.wv.vocab)
    
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in corpus]
    
    return np.array(features)


feature_size = 100

# get document level embeddings
w2v_feature_array = averaged_word_vectorizer(corpus = words_list, model = model,
                                             num_features = feature_size)
pd.DataFrame(w2v_feature_array)



#We need to also, one-hot code the target variable

data['sentiment'] = data['sentiment'].replace(['NEU'], '1')
data['sentiment'] = data['sentiment'].replace(['POS'], '2')
data['sentiment'] = data['sentiment'].replace(['NEG'], '0')

data.sentiment.value_counts()

# Transform target to an array
target = data['sentiment'].values

y = target
X = w2v_feature_array 

#Split Train and Test set
X_train, X_test, y_train, y_test =train_test_split (X, y, test_size=.30, stratify=y, random_state=(42))


#Random Forest Classifier
rfs = RandomForestClassifier(random_state=(42), n_estimators=10, n_jobs=(4))    
rfs = rfs.fit(X_train, y_train)

result = rfs.predict(X_test)

print(classification_report(y_test, result))


#Linear Vector Classifier 
svc = LinearSVC(max_iter=2000, random_state=(42))
svc = svc.fit(X_train, y_train)

result = svc.predict(X_test)

print(classification_report(y_test, result))

#Naive bayes 
gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)

result = gnb.predict(X_test)

print(classification_report(y_test, result))


##Glove Vectors 

#First we have to lead the Glove vectors in a dictionary
#We are using a Spanish Glove vector taken from: 
#https://www.kaggle.com/datasets/rtatman/pretrained-word-vectors-for-spanish


f = open('Spanish_vectors.txt', encoding="utf8")    


embeddings_index = {}

for line in tqdm(f):
    # Splitting the each line 
    values = line.split()
    
    word = values[0]
    
    coefs = np.array(values[1:], dtype='float32')
    
    embeddings_index[word] = coefs
    
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# This function creates a normalized vector for the whole sentence

def sent2vec(s):
    words = word_tokenize(s)
    N = []
    for w in words:
        try:
            N.append(embeddings_index[w])
        except:
            continue
    N = np.array(N)
    v = N.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())




#Split Train and Test set

X = data.final_text.values

X_train, X_test, y_train, y_test =train_test_split (X, y, test_size=.30, stratify=y, random_state=(42))


#Create sentence vectors using the above function for training and validations sets

X_train_glove = [sent2vec(x) for x in tqdm(X_train)]
X_test_glove = [sent2vec(x) for x in tqdm(X_test)]


#Random Forest Classifier
rfs = RandomForestClassifier(random_state=(42))    
rfs = rfs.fit(X_train_glove, y_train)

result = rfs.predict(X_test_glove)

print(classification_report(y_test, result))

#Linear Vector Classifier 
svc = LinearSVC(max_iter=2000, random_state=(42))
svc = svc.fit(X_train_glove, y_train)

result = svc.predict(X_test_glove)

print(classification_report(y_test, result))

#Naive bayes 
gnb = GaussianNB()
gnb = gnb.fit(X_train_glove, y_train)

result = gnb.predict(X_test_glove)

print(classification_report(y_test, result))

#As you could see the model that performs the best is GloVe-Random Forest


















