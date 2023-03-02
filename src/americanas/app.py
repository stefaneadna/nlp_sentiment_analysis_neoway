import streamlit as st
from pipelines import predict_pipeline
import argparse
import time
from dataset.utils import read_pre_processing_dataset
from graphs.utils import count_reviews

def main():

    ap = argparse.ArgumentParser()

    
    ap.add_argument('-pm', '--path_model',
                    default= r'../../models/lr.pkl',
                    help='Caminho do classificador treinado.')
    
    ap.add_argument('-cv', '--path_count',
                    default= r'../../models/count.pkl',
                    help='Caminho do CountVectorizer.')
    
    ap.add_argument('-tf', '--path_tfidf',
                    default= r'../../models/tfidf.pkl',
                    help='Caminho do TF-IDF.')


    args = vars(ap.parse_args())
    st.set_page_config(page_title='Americanas')

    page = st.sidebar.selectbox('Selecione a página',
                                ['Analisador de sentimento','Documentação'])
    
    df = read_pre_processing_dataset('../../dataset/B2W-Reviews01.csv')

    if page == 'Analisador de sentimento':
        st.title('Analisador de sentimento')
        title = st.text_input('Título da avaliação')
        review = st.text_area('Avaliação do produto')

        if st.button('Enviar'):
            text = title + " " + review
            pred = predict_pipeline(text, **args)
            if(pred==1):
                st.write('<span style="color:green">O Sentimento desta avaliação é positivo.</span>', unsafe_allow_html=True)
            else:
                st.write('<span style="color:red">O Sentimento desta avaliação é negativo.</span>', unsafe_allow_html=True)
    else:
        st.title('Documentação')





if __name__ == "__main__":
    main()

