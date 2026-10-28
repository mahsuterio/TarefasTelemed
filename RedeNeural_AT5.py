import pandas as pd
import torch
from pandas.api.types import is_numeric_dtype
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

df = pd.read_csv("dados/train_org.csv")

class Modelo(nn.Module):
    def __init__(self, in_features, h1=128, h2=64, h3=32, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1) # camada 1
        self.fc2 = nn.Linear(h1,h2) # camada 2
        self.fc3 = nn.Linear(h2,h3) # camada 3
        self.out = nn.Linear(h3, out_features) # output

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)

        return x

# ------------------------- FUNÇÕES --------------------------------

def metricas(precos_reais,precos_previstos,nome_modelo):
  mae = torch.mean(torch.abs(precos_previstos - precos_reais))

  rmse = torch.sqrt(torch.mean((precos_previstos - precos_reais) ** 2))

  ss_res = torch.sum((precos_reais - precos_previstos) ** 2)
  ss_tot = torch.sum((precos_reais - torch.mean(precos_reais)) ** 2)
  r2 = 1 - (ss_res / ss_tot)

  print(f"\n--- RESULTADOS {nome_modelo} ---")
  print(f"MAE: R$ {mae.item():.2f}")
  print(f"RMSE: R$ {rmse.item():.2f}")
  print(f"R²: {r2.item():.4f}")
  return mae.item(), rmse.item(), r2.item()


def separa_limpa(df): 
    train, test = train_test_split(df, test_size=0.2, random_state=42)
  
    colunas = ["Id","Utilities", "Street", "Condition2", "RoofMatl", "Heating", "PoolQC", "Exterior1st", "SaleType"]
    for c in colunas: 
        train.drop(c, axis=1, inplace=True)
        test.drop(c,axis=1, inplace=True)

    ohe = OneHotEncoder(sparse_output=False,handle_unknown='ignore').set_output(transform='pandas')

    for coluna in train.columns:
      if is_numeric_dtype(train[coluna]):
                  # pega todos os valores nao nulos e faz a media
                  # percorre todas as linhas nulas dessa coluna e altera para a media
                  media = train[coluna].mean()
                  train.loc[train[coluna].isnull() == True, coluna] = media
                  test.loc[test[coluna].isnull() == True, coluna] = media

      else:
                  # substitui valores nulos por uma categoria Nao Informado
                  train.loc[train[coluna].isnull() == True, coluna] = 'Nao Informado'
                  test.loc[test[coluna].isnull() == True, coluna] = 'Nao Informado'

      if coluna == 'SalePrice':
              train[coluna] = np.log(train[coluna])
              test[coluna] = np.log(test[coluna])
              
              # calcula a media e desvio apenas de treino para aplicar o zscore em ambos
              media_preco = train[coluna].mean()
              desvio_preco = train[coluna].std(ddof=0)

              if desvio_preco > 0:
                  train[coluna] = (train[coluna] - media_preco) / desvio_preco
                  test[coluna] = (test[coluna] - media_preco) / desvio_preco
              else:
                  train[coluna] = 0.0
                  test[coluna] = 0.0

      elif is_numeric_dtype(train[coluna]):
              media = train[coluna].mean()
              desvio = train[coluna].std(ddof=0)

              if desvio > 0:
                  train[coluna] = (train[coluna] - media) / desvio
                  test[coluna] = (test[coluna] - media) / desvio
              else:
                  train[coluna] = 0.0
                  test[coluna] = 0.0

      if not is_numeric_dtype(train[coluna]):
              # trocando o que foi solicitado nas colunas categoricas
              valores = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
              train[coluna] = train[coluna].replace(valores)
              test[coluna] = test[coluna].replace(valores)

              train[coluna] = train[coluna].astype(str) # transforma em string para o OHE
              test[coluna] = test[coluna].astype(str) # transforma em string para o OHE

              ohe_train = ohe.fit_transform(train[[coluna]])
              ohe_test = ohe.transform(test[[coluna]])

              train = pd.concat([train, ohe_train], axis=1).drop(columns=coluna)
              test = pd.concat([test, ohe_test], axis=1).drop(columns=coluna)

    X_treino = train.drop(columns=["SalePrice"])
    y_treino = train[["SalePrice"]]

    X_teste = test.drop(columns=["SalePrice"])
    y_teste = test[["SalePrice"]]

    return X_treino, X_teste, y_treino, y_teste, media_preco, desvio_preco


def treinar_modelo(modelo, learning_rate, epocas=100, grafico=False):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(modelo.parameters(),lr=learning_rate)

    losses = []

    modelo.train()

    for epoca in range(epocas):

        y_pred = modelo(tensor_X_treino)
        loss = criterion(y_pred, tensor_y_treino)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if grafico:
        plt.plot(range(epocas), losses)
        plt.ylabel("Perda")
        plt.xlabel("Epoca")
        plt.show()

    return losses


def avaliar_modelo(modelo, nome_modelo):
    modelo.eval()

    with torch.no_grad():
        previsoes = modelo(tensor_X_teste)

        precos_previstos = torch.exp(previsoes * desvio_preco + media_preco)
        precos_reais = torch.exp(tensor_y_teste * desvio_preco + media_preco)

    mae, rmse, r2 = metricas(precos_reais, precos_previstos, nome_modelo)

    return mae, rmse, r2

# tensores de treino e teste

X_treino, X_teste, y_treino, y_teste, media_preco, desvio_preco = separa_limpa(df)

tensor_X_treino = torch.tensor(X_treino.to_numpy(), dtype=torch.float32)
tensor_y_treino = torch.tensor(y_treino.to_numpy(), dtype=torch.float32)
tensor_X_teste = torch.tensor(X_teste.to_numpy(), dtype=torch.float32)
tensor_y_teste = torch.tensor(y_teste.to_numpy(), dtype=torch.float32)

# ------------------------- RESULTADOS --------------------------------

# Modelo 1 - Learning rate 0.001
torch.manual_seed(42)
modelo1 = Modelo(X_treino.shape[1])

losses1 = treinar_modelo(modelo1,learning_rate=0.001,epocas=100)

# Modelo 2 - Learning rate 0.01
torch.manual_seed(42)
modelo2 = Modelo(X_treino.shape[1])

losses2 = treinar_modelo(modelo2,learning_rate=0.01,epocas=100)

# Modelo 3 - Learning rate 0.01
torch.manual_seed(42)
modelo3 = Modelo(X_treino.shape[1])

losses3 = treinar_modelo(modelo3,learning_rate=0.001,epocas=200)

mae1, rmse1, r21 = avaliar_modelo(modelo1, "Modelo 1 - lr=0.001")
mae2, rmse2, r22 = avaliar_modelo(modelo2, "Modelo 2 - lr=0.01")
mae3, rmse3, r23 = avaliar_modelo(modelo3, "Modelo 3 - lr=0.001")