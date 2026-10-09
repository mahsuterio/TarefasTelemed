import torch
# Backpropagation: algoritmo para treino de redes neurais, calcula o erro na saída e
# o propaga de volta para a rede ajustar. "A IA aprende com os própios erros.".

# Loss Function: fórmula que mede o desvio de previsão feita por IA e o valor real
# A Loss Function calcula o tamanho do erro que será enviado para a rede neural
# por meio de Backpropagation.

# MSE (Mean Squared Error) é uma função de perda utilizada em problemas de regressão
# Exemplos de aplicação: previsão de temperaturas e de preços de imóveis
def mse(y_pred, y_real) : return torch.mean((y_pred - y_real) ** 2)

# Hinge Loss é utilizada em problemas de classificação binária
# Exemplos de aplicação: diagnóstico médico e detecção de fraudes bancárias

# L(y,y`)= max(0,1 - y . y`}
# faz isso com todos os valores e depois tira a média
# Classificações: correta com margem suficiente: perda = 0
# correta, mas com margem insuficiente: perda > 0
# incorreta: perda >= 1
def hinge_loss(y_pred, y_real) : return torch.mean(torch.clamp(1 - y_real * y_pred, min=0))

y_vdd = torch.tensor([100, 200, 300, 400, 500]).float()
y_aprox = torch.tensor([110, 190, 299, 350, 489]).float()

print("Resultado MSE:", mse(y_vdd, y_aprox))

y_vdd = torch.tensor([1.0, 1.0, 1.0])
y_aprox = torch.tensor([2.0, 0.5, -1.0])

print("Resultado HINGE:", hinge_loss(y_vdd, y_aprox))