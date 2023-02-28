from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,confusion_matrix
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s')

def train_models(X_train,X_test,y_train,y_test,model,name_model):
    """Treina um classificador e calcula as métricas: acurácia, precisão, recall, f1-score, e gera a matriz de confusão.

    Args:
        X_train ([type]): Dados para treinamento
        X_test ([type]): Dados de teste
        y_train ([type]): Rótulos de treinamento
        y_test ([type]): Rótulos de teste
        model ([type]): Classificador
        name_model ([type]): Nome do classificador

    Returns:
        dict: Retorna um dicionário com as informações: 
            Name Model, Precision, Recall, F1-Score, Accuracy, Confusion Matrix e o modelo treinado
    """
    logging.info('Treinando o Classificador')
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
