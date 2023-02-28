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
    """Remove os caracteres especiais de uma frase

    Args:
        text (String): Frase para remoção de seus caracteres especiais
    Returns:
        String: Frase após o pré-processamento
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
    """Remove as stop words de uma frase e tokeniza a frase. 

    Args:
        text (String): Frase para tokenização e remoção de stop words

    Returns:
        list: Uma lista contendo todos os tokens da frase
    """
    logging.info('Remove stop words and tokenize a sentence.')
    text = remove_special_characters(text)
    text_tokens = word_tokenize(text, language='portuguese') 
    stop_words = set(stopwords.words('portuguese'))
    text_without_sw = [token for token in text_tokens if token not in stop_words]
    return text_without_sw 

def lemmatize(text):
    """Realiza o pré-processamento de uma frase, realizando remoção de caracteres especiais e de stop words,
       faz também a tokenização e lematização da frase. 

    Args:
        text (list): A frase que deve ser pré-processada

    Returns:
        list: Uma lista com os tokens lematizados
    """
    logging.info('Lemmatize the tokens of a sentence.')
    nlp = spacy.load('pt_core_news_sm')
    text = token_and_remove_sw(text)
    doc = nlp(str(text))
    text_lemma = [token.lemma_ for token in doc 
                  if token.pos_ not in ['PUNCT','PROPN','ADP']]
    return text_lemma

def stemmize(text):
    """Realiza o pré-processamento de uma frase, realizando remoção de caracteres especiais e de stop words,
       faz também a tokenização e stemização da frase. 

    Args:
        text (list): A frase que deve ser pré-processada

    Returns:
        list: Uma lista com os tokens stemizados
    """
    logging.info('Stemmize the tokens of a sentence.')
    stemmer = RSLPStemmer()
    text = token_and_remove_sw(text)
    text_stem = [stemmer.stem(token) for token in text]
    return text_stem

def lemmatize_and_stemmize(text):
    """Realiza o pré-processamento de uma frase, realizando remoção de caracteres especiais e de stop words,
       faz também a tokenização, lematização stemização da frase. 

    Args:
        text (list): A frase que deve ser pré-processada

    Returns:
        list: Uma lista com os tokens lematizados e stemizados
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
    """Utiliza o TF-IDF para vetorizar um texto, calculando a frequencia das palavras em um texto

    Args:
        X_train (list): Uma lista contendo os dados para treinamento.
        X_test (list): Uma lista contendo os dados para teste.

    Returns:
        X_train (list): Dataset de treinamento vetorizado com o TF-IDF
        X_test (list): Dataset de teste vetorizado com o TF-IDF
    """
    vetorizer = CountVectorizer()
    X_train = vetorizer.fit_transform(X_train.map(' '.join))
    X_test = vetorizer.transform(X_test.map(' '.join))

    transformer = TfidfTransformer()
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    return X_train, X_test
