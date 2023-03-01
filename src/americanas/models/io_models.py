import pickle

def save_model(path, model):
    """Salva o modelo em um path.

    Args:
        path (String): Caminho da pasta q o modelo deve ser salvo
        model (Sklearn Model): Modelo para ser salvo
    """
    with open(path, 'wb') as file:
        pickle.dump(model, file)

def load_model(path):
    """Carrega um modelo de um path

    Args:
        path (String): Localização do model em um caminho

    Returns:
        Sklearn Model: Modelo treinado
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)    
    return model