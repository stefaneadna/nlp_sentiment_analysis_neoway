import re
from nltk import word_tokenize, download
from nltk.corpus import stopwords
import spacy
import nltk
nltk.download('rslp')
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

download('stopwords')


def remove_special_characters(text):
    """Remove os caracteres especiais de uma frase

    Args:
        text (String): Frase para remoção de seus caracteres especiais
    Returns:
        String: Frase após o pré-processamento
    """    
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
    text = remove_special_characters(text)
    text_tokens = word_tokenize(text, language='portuguese') 
    stop_words = set(stopwords.words('portuguese'))
    text_without_sw = [token for token in text_tokens if token not in stop_words]
    return text_without_sw 

def lemmatize(text):
    """Realiza o pré-processamento de uma frase, realizando remoção de caracteres especiais e de stop words,
       faz também a tokenização e lematização da frase. 

    Args:
        text (String): A frase que deve ser pré-processada

    Returns:
        list: Uma lista com os tokens lematizados
    """
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
        text (String): A frase que deve ser pré-processada

    Returns:
        list: Uma lista com os tokens stemizados
    """
    stemmer = RSLPStemmer()
    text = token_and_remove_sw(text)
    text_stem = [stemmer.stem(token) for token in text]
    return text_stem

def lemmatize_and_stemmize(text):
    """Realiza o pré-processamento de uma frase, realizando remoção de caracteres especiais e de stop words,
       faz também a tokenização, lematização stemização da frase. 

    Args:
        text (String): A frase que deve ser pré-processada

    Returns:
        list: Uma lista com os tokens lematizados e stemizados
    """
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
        X_train (array): Uma lista contendo os dados para treinamento.
        X_test (array): Uma lista contendo os dados para teste.

    Returns:
        X_train (array): Dataset de treinamento vetorizado com o TF-IDF
        X_test (array): Dataset de teste vetorizado com o TF-IDF
    """
    vetorizer = CountVectorizer()
    X_train = vetorizer.fit_transform(X_train.map(' '.join))
    X_test = vetorizer.transform(X_test.map(' '.join))

    transformer = TfidfTransformer()
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    return X_train, X_test


def scaler(X_train,X_test):
    """Utiliza o StandardScaler para normalizar os dados.

    Args:
        X_train (array): Uma lista contendo os dados para treinamento.
        X_test (array): Uma lista contendo os dados para teste.

    Returns:
        X_train (array): Dataset de treinamento normalizados.
        X_test (array): Dataset de teste normalizados.
    """
    scaler = StandardScaler(with_mean=False).fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test
