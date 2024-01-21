import streamlit as st
import pandas as pd 
import numpy as np
import tensorflow_hub as tf_hub
from tensorflow.keras.models import load_model
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

model = load_model('best_model.h5', custom_objects={'KerasLayer': tf_hub.KerasLayer})

# Create a dictionary to map the labels to the categories
label_dict = {0: 'business', 1: 'tech', 2: 'politics', 3: 'sport', 4: 'entertainment'}

def preprocessing(text):
    ''' 
    Preprocessing text by applying lowercasing, normalization, tokenization, stopword removal, and lemmatization
    '''
    # Lowercase the text
    text = text.lower()

    # Normalize the text
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove whitespaces

    # Tokenize the text
    tokens = word_tokenize(text)

    # Get the English stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update(['also', 'said', 'would', 'could']) # add some frequent words that won't help the model

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize the text
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Combine tokens back into a single string
    text = ' '.join(tokens)

    return text

def run():
    st.title('News Categorization')

    default = """New Film Showcases One Year of Unprecedented Creativity

    In a year that has seen the world grapple with unprecedented challenges, one film has emerged as a beacon of hope and resilience. The new show, aptly titled "One Year", is set to premiere this weekend, promising to take viewers on a cinematic journey like no other.

    "One Year" is not just a film; it's a testament to the indomitable spirit of creativity that thrives even in the most trying times. The film chronicles the experiences of a diverse group of artists over the course of a year, capturing their struggles, triumphs, and the extraordinary art they produce against all odds.

    The new show is already generating buzz in the entertainment industry, with critics praising its raw, unfiltered portrayal of the artistic process. The film's unique narrative structure, which unfolds over the course of a year, offers a fresh perspective on the timeless themes of perseverance and the transformative power of art.

    As the world eagerly awaits the premiere of "One Year", there's no doubt that this film is set to be one of the most talked-about shows of the year. So, mark your calendars and prepare to be inspired by this captivating new show that celebrates the enduring spirit of creativity. Stay tuned for more updates!
    """

    user_input = st.text_area("Enter the news text here:", default, height=500)

    if st.button('Predict'):
        # Apply the function to the 'Text' column in the data
        text_processed = preprocessing(user_input)

        # The model expects input data in batch, even if just predicting on one sample
        # So, I'll add an extra dimension with np.expand_dims
        preprocessed_article = np.expand_dims(text_processed, axis=0)

        # get the prediction
        predictions = model.predict(preprocessed_article)

        # get the class with the highest probability
        predicted_class = np.argmax(predictions[0])

        # Decode the predicted class into the original category
        predicted_category = label_dict[predicted_class]

        st.write(f'The predicted category is: {predicted_category}')

if __name__ == '__main__':
    main()