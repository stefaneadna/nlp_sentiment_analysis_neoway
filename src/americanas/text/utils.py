import re
from nltk import word_tokenize, download
from nltk.corpus import stopwords
import spacy
import nltk
nltk.download('rslp')
nltk.download('punkt')
from nltk.stem import RSLPStemmer
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

