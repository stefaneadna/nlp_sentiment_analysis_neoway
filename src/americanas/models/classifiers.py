from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,confusion_matrix
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import BernoulliNB


def train_models(X_train,X_test,y_train,y_test,model,name_model):
    """Treina um classificador e calcula as métricas: acurácia, precisão, recall, f1-score, e gera a matriz de confusão.

    Args:
        X_train (array): Dados para treinamento
        X_test (array): Dados de teste
        y_train (array): Rótulos de treinamento
        y_test (array): Rótulos de teste
        model (sklearn model): Classificador
        name_model (string): Nome do classificador

    Returns:
        dict: Retorna um dicionário com as informações: 
            Name Model, Precision, Recall, F1-Score, Accuracy, Confusion Matrix e o modelo treinado
    """
    model.fit(X_train, y_train)    
    predictions = model.predict(X_test)

    #Calcula as metricas do modelo
    acc = accuracy_score(y_test, predictions) * 100
    precision = precision_score(y_test, predictions) * 100
    recall = recall_score(y_test, predictions) * 100
    f1 = f1_score(y_test, predictions) * 100
    conf_mat = confusion_matrix(y_test, predictions)
    metrics = {
        'Name Model':name_model,
        'Precision':precision,
        'Recall': recall,
        'F1-score': f1,
        'Accuracy': acc,
        'Confusion matrix':conf_mat,
        'Model': model
    }
    return metrics

def logistic_regression():
    """Cria e configura os parâmetros para um modelo de regressão logistica.

    Returns:
        Sklearn model: Um modelo de regressão logística.
    """
    model = LogisticRegression(max_iter=3000, 
                               solver='liblinear', 
                               multi_class='ovr', 
                               random_state=12345)
    return model 

def bernoullinb():
    """Cria e configura os parâmetros para o modelo Bernoulli.

    Returns:
        Sklearn model: Um modelo de Bernoulli.
    """
    model = BernoulliNB()
    return model 

def knn():
    """Cria e configura os parâmetros para o modelo KNN.

    Returns:
        Sklearn model: Um modelo KNN.
    """
    model = KNeighborsClassifier(n_neighbors= 3)
    return model

def random_forest():
    """Cria e configura os parâmetros para o modelo Random Forest

    Returns:
        Sklearn model: Um modelo Random Forest,
    """
    model = RandomForestClassifier(criterion= 'gini', 
                                   max_depth= 8, 
                                   max_features= 'sqrt', 
                                   min_samples_split= 2, 
                                   n_estimators= 200, 
                                   random_state=12345)
    return model

def bagging_classifier():
    """Cria e configura os parâmetros para o modelo que utiliza a técnica de Bagging com uma regressão logística.

    Returns:
        Sklearn model: Um modelo Bagging com regressão logística.
    """
    model = BaggingClassifier(LogisticRegression(max_iter=3000, 
                                         solver='liblinear', 
                                         multi_class='ovr', 
                                         random_state=12345))
    return model