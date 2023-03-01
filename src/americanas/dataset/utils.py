import pandas as pd

def read_pre_processing_dataset(path):
    """carrega um dataset e realiza o pré processamento dele, removendo dados nulos e duplicados. 

    Args:
        path (String): Caminho para o arquivo csv referente ao dataset.

    Returns:
        Dataframe: Um dataset pandas pré processado.
    """
    df = pd.read_csv(path)
    df.dropna(subset=['review_title', 'overall_rating','recommend_to_a_friend','review_text','reviewer_birth_year'],inplace=True)
    df['reviewer_birth_year'] = df['reviewer_birth_year'].astype('int')
    df['recommend_to_a_friend_cat'] = df['recommend_to_a_friend'].map({'Yes':1,'No':0})
    df.drop_duplicates(inplace=True)
    df = df.reset_index()
    df['title_text_review'] = [title+" "+text for title, text in zip(df['review_title'],df['review_text'])]
    return df