import torch

# Learnin rate refere-se a um hiperparâmetro que mede o tamanho dos passos que um modelo 
# de machine learning dá quando seu peso é atualizado durante o processo de otimização
# 0.1 alto, 0.01 mais comum, 0.001 redes neurais maiores (Adam).

# Otimizador é um algoritmo que ajusta os biases e pesos da rede neural, com o objetivo de minimizar
# a loss function e melhorar o resultado do modelo. Ele se relaciona com a learning rate uma vez que
# ela define o "tamanho do passo", podendo manter a taxa fixa ou ajustável, e o otimizador deve 
# analisar isso.

# A diferença está em como eles calculam os pasos e ajustes de pesos da rede neural:
# SGD tem a learning rate estatica, Adam tem a learning rate ajustável.

# O batch é um processamento em lote, onde dados são agrupados e executados de uma vez
# é utilizado para economizar recursos, melhor organização de grandes volumes e execução
# fluida em horários de pico.

# Uma função de ativação é uma operação matemática aplicada em uma rede neural, em que sua função
# é decidir se um neurônio deve estar ativo ou permanecer inativo

tensor = torch.tensor([
    [-12,  45,   0,  -8,  92, -31,  17,   0, -65,  23, -4,  81],
    [100, -23,  56,   0, -91,  34, -17,  72,   0, -48, 63,  -5],
    [-99,  12, -45,  88,   0, -73,  51, -16,  29,   0, -8,  67],
    [ 34,   0, -81,  25, -14,  93, -56,  11, -37,  68,  0, -22],
    [ -7, -55,  43, -19,  76,   0, -32,  85, -64,  21, -9,  10],
    [  0,  98, -12,  47, -88,  33,   0, -25,  61, -49, 74, -16],
    [-43,  27,   0, -69,  14,  82, -11,  39, -95,   0, 58, -30],
    [ 71, -18,  35,   0, -52,  96, -24,   8, -77,  42,  0, -13],
    [-26,  59, -83,  15,   0, -41,  73, -20,  54, -67,  9,  31],
    [ 87,   0, -34,  62, -15,  28, -90,  46,   0, -53, 19, -72]
])

# Implemente uma função que recebe um tensor e seu output é a função de ativação ReLU com suporte a batches

def relu(tensor):
  condicao = (tensor>0).int()
  linhas, colunas = tensor.shape
  novo_tensor = tensor.clone()

  for i in range(linhas):
    for j in range(colunas):

      if condicao[i][j] == 0:
        novo_tensor[i][j] = 0
  
  return novo_tensor

def ativacao(tensor) : return relu(tensor)

print(tensor,'\n')
print(ativacao(tensor),'\n')

y_pred = torch.tensor([
    [2.0, 4.0, 6.0],
    [3.0, 5.0, 7.0]
])

y_real = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 4.0, 6.0]
])

# Adicione suporte a batches para uma loss function da penúltima atividade

def mse(y_pred,y_real,dim=None): 
  erro = (y_pred - y_real) ** 2

  # com dim=1, retorna o mse de cada linha. com dim=0, retorna o mse de cada coluna. dim=None, retorna o mse global
  return torch.mean(erro,dim=dim) 

print(mse(y_pred,y_real),'\n')
print(mse(y_pred,y_real,0),'\n')
print(mse(y_pred,y_real,1),'\n')
