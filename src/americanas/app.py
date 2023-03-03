import streamlit as st
from pipelines import predict_pipeline
import argparse
import time
from dataset.utils import read_pre_processing_dataset
from PIL import Image

def main():

    ap = argparse.ArgumentParser()

    
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
    st.set_page_config(page_title='Americanas')

    page = st.sidebar.selectbox('Selecione a página',
                                ['Analisador de sentimento','DashBoard'])
    
    df = read_pre_processing_dataset('dataset/B2W-Reviews01.csv')

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

        len_data = len(df)
        tit,metr1,metr2= st.columns([.4,.3,.3])
        tit.title('DashBoard')
        len_pos = (len(df[df['recommend_to_a_friend']=='Yes']) / len_data) * 100
        len_neg = (len(df[df['recommend_to_a_friend']=='No']) / len_data) * 100
        metr1.metric(label="Avaliações Positivas", value=f'{len_pos:.2f}%')
        metr2.metric(label="Avaliações Negativas", value=f'{len_neg:.2f}%')

    
        col1,col2 = st.columns([.7,.3])

        categoria = Image.open('imagens/categoria.png')
        col1.subheader('Avaliações por categoria de produto')
        col1.image(categoria)

        col2.subheader('WordCloud')
        aval_boas = Image.open('imagens/aval_boas.png')
        aval_ruins = Image.open('imagens/aval_ruins.png')

        col2.image(aval_boas)
        col2.image(aval_ruins)

        st.subheader('Sentimento das avaliações por produto')
        col3,col4 = st.columns([.5,.5])
        pos = Image.open('imagens/pos.png')
        neg = Image.open('imagens/neg.png')
        col3.image(pos)
        col4.image(neg)

        st.subheader('Análise do gênero e idade dos usuários')
        col5,col6 = st.columns([.6,.4])
        idade = Image.open('imagens/idade.png')
        genero = Image.open('imagens/genero.png')
        col5.image(idade)
        col6.image(genero)

    

if __name__ == "__main__":
    main()

