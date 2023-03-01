import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.utils import read_pre_processing_dataset
from text.utils import lemmatize
from models.vectorizer import vetorizer_tfidf
from sklearn.linear_model import LogisticRegression
from models.classifiers import train_models, logistic_regression
from models.io_models import save_model
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s')


def train_pipeline(path_dataset,path_model,path_count,path_tfidf):
    """Pipeline para o treinamento do classificador

    Args:
        path_dataset (String): Caminho do dataset.
        path_model (String): Caminho para salvar o classificador. 
        path_count(String): Caminho para salvar o CountVectorizer.
        path_tfidf(String): Caminho para salvar o TF-IDF
    """
    try:
        logging.info(f'Carregando o dataset do path:{path_dataset}')
        df = read_pre_processing_dataset(path_dataset)
        df['review_lemma'] = df['title_text_review'].apply(lemmatize)
    except:
        logging.error(f'O path indicado não existe: {path_dataset}')
        exit()


    X = df['review_lemma']
    y = df['recommend_to_a_friend_cat']

    logging.info(f'Dividindo os dados em treino e teste..')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42,stratify=y)

    logging.info(f'Realizando pré-processamento textual..')
    X_train, X_test = vetorizer_tfidf(X_train,X_test,path_count,path_tfidf)

    logging.info(f'Treinando o classificador..')
    name_model = "Logistic Regression"
    model = logistic_regression()
    metrics_model = train_models(X_train, X_test, y_train, y_test,model, name_model)

    try:
        logging.info(f'Salvando o modelo no path: {path_model}')
        save_model(path_model, metrics_model['Model'])
    except:
        logging.error(f'O path indicado não existe: {path_model}')
        exit()
    
    logging.info(f"Métricas:\n-Acurácia:{metrics_model['Accuracy']:.2f}\n-F1Score:{metrics_model['F1-score']:.2f}\n-Precisão:{metrics_model['Precision']:.2f}\n-Recall:{metrics_model['Recall']:.2f}")