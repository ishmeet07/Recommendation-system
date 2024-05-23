import subprocess
import nltk

#nltk.download('all')

# Download the punkt resource
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')




# Rest of your code...

# Rest of your code...
import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image

# Load the dataset
data = pd.read_csv('amazon_product.csv')

# Remove unnecessary columns
data = data.drop('id', axis=1)

# Define tokenizer and stemmer
stemmer = SnowballStemmer('english')
def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Create stemmed tokens column
data['stemmed_tokens'] = data.apply(lambda row: tokenize_and_stem(row['Title'] + ' ' + row['Description']), axis=1)

# Define TF-IDF vectorizer and cosine similarity function
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem)
def cosine_sim(text1, text2):
    text1_concatenated = ' '.join(text1)
    text2_concatenated = ' '.join(text2)
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1_concatenated, text2_concatenated])
    return cosine_similarity(tfidf_matrix)[0][1]

# Define search function
def search_products(query):
    query_stemmed = tokenize_and_stem(query)
    data['similarity'] = data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))
    results = data.sort_values(by=['similarity'], ascending=False).head(10)[['Title', 'Description', 'Category']]
    return results

# Rest of your code...
page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
img = Image.open('img.png')
st.image(img, width=900)
st.title("Product Recommendation Engine")
query = st.text_input("Enter Product Name")
submit = st.button('Search for similar products')
st.text('Team HypeHunter - Minor Project')
st.text('Members - Ashpreet Singh , Amrit Gaur, Ishmeet Singh, Jashan Pal Singh')
if submit:
    res = search_products(query)
    st.write(res)
