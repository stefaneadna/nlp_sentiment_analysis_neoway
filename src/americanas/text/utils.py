import re
from nltk import word_tokenize, download
from nltk.corpus import stopwords
import spacy
import nltk
nltk.download('rslp')
from nltk.stem import RSLPStemmer
import logging
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer

download('stopwords')
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s')

def remove_special_characters(text):
    """Remove special characters from a sentence.

    Args:
        text (String): Sentence to remove special characters.
    Returns:
        String: Pre-processed sentence.
    """    
    logging.info('Removing special characters')
    characters_1 = re.compile("[$.;:!\'?@,\"()\[\]]")
    characters_2 = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    characters_3 = re.compile('[0-9]+')

    text = characters_1.sub("", text)
    text = characters_2.sub(" ", text.lower())
    text = characters_3.sub("", text)
    
    return text

def token_and_remove_sw(text):
    """Remove stop words and tokenize a sentence.

    Args:
        text (String): Sentece to Remove stop words and tokenize.

    Returns:
        list: A list of all phrase tokens.
    """
    logging.info('Remove stop words and tokenize a sentence.')
    text = remove_special_characters(text)
    text_tokens = word_tokenize(text, language='portuguese') 
    stop_words = set(stopwords.words('portuguese'))
    text_without_sw = [token for token in text_tokens if token not in stop_words]
    return text_without_sw 

def lemmatize(text):
    """Lemmatize the tokens of a sentence.

    Args:
        text (list): A list of tokens

    Returns:
        list: A list of lemmatized tokens
    """
    logging.info('Lemmatize the tokens of a sentence.')
    nlp = spacy.load('pt_core_news_sm')
    text = token_and_remove_sw(text)
    doc = nlp(str(text))
    text_lemma = [token.lemma_ for token in doc 
                  if token.pos_ not in ['PUNCT','PROPN','ADP']]
    return text_lemma

def stemmize(text):
    """Stemmize the tokens of a sentence.

    Args:
        text (list): A list of tokens

    Returns:
        list: A list of stemmed tokens
    """
    logging.info('Stemmize the tokens of a sentence.')
    stemmer = RSLPStemmer()
    text = token_and_remove_sw(text)
    text_stem = [stemmer.stem(token) for token in text]
    return text_stem

def lemmatize_and_stemmize(text):
    """Lemmatize and stemmize the tokens of a sentence.

    Args:
        text (list): A list of tokens

    Returns:
        list: A list of lemmatized and stemmed tokens
    """
    logging.info('Lemmatize and stemmize the tokens of a sentence.')
    nlp = spacy.load('pt_core_news_sm')
    stemmer = RSLPStemmer()
    text = token_and_remove_sw(text)
    doc = nlp(str(text))
    text_lemma_stem = [stemmer.stem(token.lemma_) for token in doc 
                       if token.pos_ not in ['PUNCT','PROPN','ADP']]
    return text_lemma_stem

def vetorizer_tfidf(X_train,X_test):
    """Calculates the frequency of a word in a document using TF-IDF

    Args:
        X_train (list): A list with the training dataset
        X_test (list): A list with the testing dataset

    Returns:
        X_train (list): Train dataset vetorized with TF-IDF
        X_test (list): Test dataset vetorized with TF-IDF
    """
    vetorizer = CountVectorizer()
    X_train = vetorizer.fit_transform(X_train.map(' '.join))
    X_test = vetorizer.transform(X_test.map(' '.join))

    transformer = TfidfTransformer()
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    return X_train, X_test
