import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
import tensorflow_datasets as tfds
import re
from nltk.tokenize import word_tokenize as wt
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import io
import sklearn
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import streamlit as st
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('LSTM_retrained.h5')
tok=joblib.load('tokenizer_retrained.pickle')

max_words = 1000
max_len = 150
#tok = Tokenizer(num_words=max_words)


def lemma(test_pred):
    lemmatizer = WordNetLemmatizer()
    test_data_lemma = []
    for i in range(test_pred.shape[0]):
        sp = test_pred.iloc[i]
        sp = re.sub('[^A-Za-z]', ' ', sp)
        sp = sp.lower()
        tokenized_sp = word_tokenize(sp)
        sp_processed = []
        for word in tokenized_sp:
            if word not in set(stopwords.words('english')):
                sp_processed.append(lemmatizer.lemmatize(word))
        sp_text = " ".join(sp_processed)
        test_data_lemma.append(sp_text)
    return pd.Series(test_data_lemma)


# Define the Streamlit application
def main():
    st.title('SpyEye-An Email Monitoring System to detect Spam')
    st.write('Enter some input data to get a prediction:')
    input_email = st.text_area('Input data:')
    if not input_email:
        st.warning('Please enter some input data.')
    else:
        test_pred = pd.Series([input_email])
        test_pred = test_pred.apply(lambda x: ' '.join([word for word in x.split() if word not in set(stopwords.words('english'))]))
        test_pred = test_pred.str.replace('\W', ' ', regex=True)
        test_pred = test_pred.str.lower()
        test_pred = lemma(test_pred)

       # tok.fit_on_texts(test_pred)
        #sequences = tok.texts_to_sequences(test_pred)
        #sequences_matrix = pad_sequences(sequences,maxlen=max_len)
        txts = tok.texts_to_sequences(test_pred)
        txts = pad_sequences(txts, maxlen=max_len)
        prediction = model.predict(txts)[0][0]

       

        if st.button('Predict'):
            #pred_proba = model.predict_proba(test_data_transformed)[0]
            #pred_label = model.predict(test_data_transformed)[0]
            #prediction = model.predict(test_data_transformed)
            #st.write('Prediction:', prediction)
            if (txts < 0.5).all():
                st.write('This email is not Spam: ',prediction)
            else:
                st.write('This email is Spam: ',prediction.round)
            

              

if __name__ == '__main__':
    main()
