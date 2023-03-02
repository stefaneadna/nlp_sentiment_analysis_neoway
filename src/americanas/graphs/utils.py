import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def matrix(matrix):
    """Plota uma matriz de confusão

    Args:
        matrix (array): Um Array referente a uma matriz de confusão de 2x2
    """
    plt.figure(figsize=(5,5))
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, fmt="", cmap="Blues")
    ax.set_xticklabels(['Ruim', 'Bom'])
    ax.set_yticklabels(['Ruim', 'Bom'])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Reviews")


def count_reviews(data):
    data1 = data[['product_name','site_category_lv1','site_category_lv2']]
    data = data['product_name'].value_counts().to_frame().reset_index()
    data.rename(columns={'index':'product_name','product_name':'Number of reviews'},inplace=True)
    data = pd.merge(
        data,
        data1[["product_name", "site_category_lv1",'site_category_lv2']],
        how="left",
        on="product_name",
    )
    data = data.drop_duplicates()
    data.rename(columns={'product_name':'Product Name',
                         'site_category_lv1':'Category 1',
                         'site_category_lv2':'Category 2'},inplace=True)
    return data