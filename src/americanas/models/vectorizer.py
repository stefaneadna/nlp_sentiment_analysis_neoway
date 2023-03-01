from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import logging
from .io_models import save_model

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s')


def vetorizer_tfidf(X_train,X_test,path_count,path_tfidf):
    """Utiliza o TF-IDF para vetorizar um texto, calculando a frequencia das palavras em um texto.

    Args:
        X_train (array): Uma lista contendo os dados para treinamento.
        X_test (array): Uma lista contendo os dados para teste.
        path_count(String): Caminho para salvar o CountVectorizer.
        path_tfidf(String): Caminho para salvar o TF-IDF

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

    try:
        logging.info(f'Salvando o CountVectorizer e o TF-IDF nos paths:\n- {path_count}\n-{path_tfidf}')
        save_model(path_count, vetorizer)
        save_model(path_tfidf, transformer)
    except:
        logging.error(f'O path indicado não existe')
        exit()

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
