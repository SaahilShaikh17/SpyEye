import pandas as pd
import joblib
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Load the machine learning model
model = joblib.load('naive_bayes_classifier.pkl')

# Load the vectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Define the lemma function
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

        # Transform the input email using the vectorizer
        test_data_transformed = vectorizer.transform(test_pred)

        if st.button('Predict'):
            pred_proba = model.predict_proba(test_data_transformed)[0]
            pred_label = model.predict(test_data_transformed)[0]
            #prediction = model.predict(test_data_transformed)
            #st.write('Prediction:', prediction)
            if pred_label == 'ham':
                st.write('This email is not Spam')
            else:
                st.write('This email is Spam')
            st.write(f"Predicted probability of spam: {pred_proba[1]:.3f}")
            st.write(f"Predicted probability of not spam: {pred_proba[0]:.3f}")

              

if __name__ == '__main__':
    main()
