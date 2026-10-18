import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import stats
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("dados/train.csv")

#deletando as colunas solicitadas
colunas = ["Utilities", "Street", "Condition2", "RoofMatl", "Heating", "PoolQC", "Exterior1st", "SaleType"]
for c in colunas : df.drop(c, axis=1, inplace=True)

ohe = OneHotEncoder(sparse_output=False).set_output(transform='pandas') # criando o ohe

for coluna in df.columns:
    nulo = df[coluna].isnull().sum()
    if nulo > 0:
        if is_numeric_dtype(df[coluna]):
            # pega todos os valores nao nulos e faz a media
            # percorre todas as linhas nulas dessa coluna e altera para a media

            media = df[coluna].mean()
            df.loc[df[coluna].isnull() == True, coluna] = media

        else:
            # substitui valores nulos por uma categoria Nao Informado

            df.loc[df[coluna].isnull() == True, coluna] = 'Nao Informado'

    if coluna == 'SalePrice':
         # trocando valores de SalePrice para o ln(zscore)
         df[coluna] = np.log(df[coluna])    
         z_score = stats.zscore(df[coluna])
         df[coluna] = z_score

    elif is_numeric_dtype(df[coluna]):
        # trocando valores de colunas numericas para o zscore
        z_score = stats.zscore(df[coluna])
        df[coluna] = z_score

    if not is_numeric_dtype(df[coluna]):
        # trocando o que foi solicitado nas colunas categoricas
        valores = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
        df[coluna] = df[coluna].replace(valores)

        df[coluna] = df[coluna].astype(str) # transforma em string para o OHE

        ohetransform = ohe.fit_transform(df[[coluna]])
        df = pd.concat([df, ohetransform], axis=1).drop(columns=coluna)

df.to_csv("dados/train.csv", index=False)