import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


data=pd.read_csv('Emotion_final.csv')


# Étape 3 : Prétraitement des données
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_data(text):
    # Tokenisation
    tokens = nltk.word_tokenize(text.lower())
    
    # Suppression de la ponctuation
    tokens = [token for token in tokens if token.isalnum()]
    
    # Suppression des stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatisation
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Reconstitution du texte prétraité
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

data['preprocessed_text'] = data['Text'].apply(preprocess_data)

# Étape 4 : Création du Bag of Words
vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(data['preprocessed_text'])
feature_names = vectorizer.get_feature_names_out()









