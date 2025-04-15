from email import message
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('wordnet')
nltk.download('stopwords')
nb = pickle.load(
    open('model.pkl', 'rb'))
v=pickle.load(open('clf.pkl', 'rb'))

def predict_spam(message):
    
    # Replace addresses (hhtp, email), numbers (plain, phone), money symbols
    message = message.replace(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b',' ')
    message = message.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)',' ')
    message = message.replace(r'£|\$', ' ')    
    message = message.replace(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',' ')    
    message = message.replace(r'\d+(\.\d+)?', ' ')

    # Remove punctuation, collapse all whitespace (spaces, line breaks, tabs) into a single space & eliminate any leading/trailing whitespace.
    message = message.replace(r'[^\w\d\s]', ' ')
    message = message.replace(r'\s+', ' ')
    message = message.replace(r'^\s+|\s+?$', '')

    
    # Lowercase the entire corpus
    message = message.lower()

    stop_words=nltk.corpus.stopwords.words('english')
    def st_word(x):
        message1=""
        for term in x.split():
            if term not in set(stop_words):
                message1+=(term+" ")
        return message1
    message=st_word(message)
    lemmatizer=nltk.stem.WordNetLemmatizer()
    def lem_word(x):
        message2=""
        for term in x.split():
            lemmatizer.lemmatize(term,pos='v')
            message2+=(term+" ")
        return message2
    message=lem_word(message)

    porter=nltk.PorterStemmer()
    def stem_word(x):
        message3=""
        for term in x.split():
            porter.stem(term)
            message3+=(term+ " ")
        return message3
    message=stem_word(message)
    print(message)
    sms=[]
    sms.append(message)
    sms_count=v.transform(sms)
    res1=nb.predict(sms_count)

    return res1


def main():
    st.title("SMS Spam Detection")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">SMS Spam Detection ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    message = st.text_input("Message", "Type Here")
    if st.button("Predict"):
        output = predict_spam(message)
        if(output[0]==0):
            st.success(f"The given message is Ham")
        else:
            st.error("The given message is Spam!!")


if __name__ == '__main__':
    main()
