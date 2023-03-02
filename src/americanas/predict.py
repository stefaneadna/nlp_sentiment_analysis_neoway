from pipelines import predict_pipeline
import argparse

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument('-tx', '--text',
                    default= r'Este produto é muito ruim.',
                    help='Texto para analise do sentimento.')

    ap.add_argument('-pm', '--path_model',
                    default= r'models/lr.pkl',
                    help='Caminho do classificador treinado.')
    
    ap.add_argument('-cv', '--path_count',
                    default= r'models/count.pkl',
                    help='Caminho do CountVectorizer.')
    
    ap.add_argument('-tf', '--path_tfidf',
                    default= r'models/tfidf.pkl',
                    help='Caminho do TF-IDF.')


    args = vars(ap.parse_args())

    pred = predict_pipeline(**args)


if __name__ == "__main__":
    main()
