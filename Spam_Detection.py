
# Commented out IPython magic to ensure Python compatibility.

#importing required libraries
import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

import re
from nltk.tokenize import word_tokenize as wt
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import io
import sklearn

import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
# %matplotlib inline

import streamlit as st
st.title('SPAM-HAM-DETECTION')
#st.button("LSTM")



df = pd.read_csv('spam_ham_dataset.csv',delimiter=',',encoding='latin-1')
df.head()

X = df.text
Y = df.label_num
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)



from tensorflow.keras.preprocessing.sequence import pad_sequences
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = pad_sequences(sequences,maxlen=max_len)


def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model


model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])


model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = pad_sequences(test_sequences,maxlen=max_len)

accr = model.evaluate(test_sequences_matrix,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


text = st.text_area("Enter email")
st.header("SPAM OR HAM Result")
if st.button("Run LSTM"):
    txts = tok.texts_to_sequences([text])
    txts = pad_sequences(txts, maxlen=max_len)
    preds = model.predict(txts)
    if preds > 0.5:
        st.success("SPAM")
        st.success(preds)
    else:
        st.success("HAM")
        st.success(preds)