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


    args = vars(ap.parse_args())

    train_pipeline(**args)



if __name__ == "__main__":
    main()




