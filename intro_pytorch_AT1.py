import torch 

# O PyTorch é uma biblioteca do python que 
# serve para criar, treinar e implantar 
# modelos de inteligência artificial, 
# machine learning e deep learning

# Um tensor é uma estrutura de dados parecida com arrays 
# e matrizes. São usados como base fundamental 
# para armazenar e manipular dados, dentro do contexto de
# Inteligencia Artificial.

# criar 2 tensores aleatorios e fazer a mult matricial deles
tensor1 = torch.rand(3,4)
tensor2 = torch.rand(4)

print(tensor1, "\n")
print(tensor2, "\n")
print(torch.matmul(tensor1, tensor2), "\n")

print("-----------------------------------------------------\n")

# criar uma função que recebe duas tuplas, cria dois
# tensores naquele shape e retorna a multiplicação
# matricial entre eles

def cria_tensor_mult(tupla1, tupla2): # criação da função
    tensor1 = torch.rand(tupla1)
    tensor2 = torch.rand(tupla2)
    return torch.matmul(tensor1, tensor2)

print(cria_tensor_mult((3,4),(4))) # chamada da função