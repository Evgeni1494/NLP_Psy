import nltk
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from elasticsearch import Elasticsearch
from faker import Faker



# Connexion à Elasticsearch
es = Elasticsearch(hosts=["http://localhost:9200"])

# Mapping de l'index "notes"
mapping = {
    "mappings": {
        "properties": {
            "patient_lastname": {"type": "keyword"},
            "patient_firstname": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "standard"},
            "date": {"type": "date"},
            "patient_left": {"type": "boolean"},
            "emotion": {"type": "keyword"},
            "confidence": {"type": "float"}
        }
    }
}

# Création de l'index avec le mapping
index_name = "notes5"
es.indices.create(index=index_name, body=mapping)


df = pd.read_csv("Predicted_Data.csv")
fake = Faker()

# Alimentation de l'index "notes"
for index, row in df.iterrows():
    doc = {
        "patient_lastname": fake.last_name(),
        "patient_firstname": fake.first_name(),
        "text": row["Text"],
        "emotion": row["Emotion"],
        "date": fake.date(),
        "patient_left": fake.boolean(),
        "confidence": row['Confidence']
    }
    es.index(index=index_name, body=doc)
es.transport.close()










