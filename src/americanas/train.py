from pipelines import train_pipeline
import argparse

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument('-pd', '--path_dataset',
                    default=r"dataset/B2W-Reviews01.csv",
                    help='Caminho do conjunto de dados das Americanas.')
    
    ap.add_argument('-pm', '--path_model',
                    default= r'models/lr.pkl',
                    help='Caminho para salvar o classificador treinado.')
    
    ap.add_argument('-cv', '--path_count',
                    default= r'models/count.pkl',
                    help='Caminho para salvar o CountVectorizer.')
    
    ap.add_argument('-tf', '--path_tfidf',
                    default= r'models/tfidf.pkl',
                    help='Caminho para salvar o TF-IDF.')


    args = vars(ap.parse_args())

    train_pipeline(**args)



if __name__ == "__main__":
    main()




