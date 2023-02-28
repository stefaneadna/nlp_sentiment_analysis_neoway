from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s')

class Loss(CallbackAny2Vec):
    """Utilizada para fazer o callback do Doc2vec

    Args:
        CallbackAny2Vec (callbacks)
    """
    def __init__(self, num_epochs):
        """Metodo construtor da classe Loss

        Args:
            num_epochs (int): Número de epocas para treinamento do modelo
        """
        self.verbose = tqdm(range(num_epochs), desc='Training Doc2Vec')
        self.epoch = 1
        self.loss = []
        self.loss_prev_step = None

    def on_epoch_end(self, model):
        """ Imprime o verbose

        Args:
            model (doc2vec): Modelo de treinamento do Doc2vec
        """
        if self.epoch == 1:
            current_loss = model.get_latest_training_loss()
        else:
            current_loss = model.get_latest_training_loss() - self.loss_previous_step
        
        self.loss.append(current_loss)
        self.loss_previous_step = model.get_latest_training_loss()
        self.epoch += 1

        self.verbose.set_postfix(loss=current_loss)
        self.verbose.update(1)


def fit_doc2vec(texts,embedding_size, num_iter=int(5e+3)):
    """Treina o modelo Doc2Vec

    Args:
        texts (array): Dataset textual para treinamento.
        embedding_size (int): Tamanho do embedding.
        num_iter (int, optional): Numero de epocas para treinamento do modelo. Defaults to int(5e+3).

    Returns:
        doc2vec: Modelo Doc2Vec treinado.
    """
    logging.info('Treinando o Doc2Vec')
    texts = [TaggedDocument(text, tags=[i]) for i,text in enumerate(texts)]
    loss = Loss(num_iter)
    model = Doc2Vec(
        vector_size=embedding_size,
        window=5,
        min_count=1,
        min_alpha=1e-4,
        epochs=num_iter,
        callbacks=[loss]
    )
    model.build_vocab(texts)
    model.train(texts, total_examples=model.corpus_count, epochs=model.epochs,callbacks = [loss] )
    return model