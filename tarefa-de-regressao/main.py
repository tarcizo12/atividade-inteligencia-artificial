import numpy as np
import matplotlib.pyplot as plt

#1. visualização dos dados com um gráfico de dispressão 
Data = np.loadtxt("base-de-dados/Ice_cream selling data.csv",delimiter=',', skiprows = 1)
temperatura = Data[:, 0]  # Armazena a primeira coluna em coluna1
vendas_sorvete = Data[:, 1]


#2. variaveis regressoras sejam armazenadas em uma matriz de dimensão N×p
##  O mesmo para o vetor de varaiveis observadas, organizando em um vetor de dimensão N×1

X = np.vstack((temperatura)).T
y = np.array(vendas_sorvete).reshape(-1, 1)


#1. visualização dos dados com um gráfico de dispressão
plt.scatter(temperatura, vendas_sorvete, label='Dados de Vendas de Sorvete')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vendas de Sorvete (unidades)')
plt.title('Relação entre Temperatura e Vendas de Sorvete')
plt.legend()
plt.grid(True)

print("Variavel de regressão: ", X , '\n')
print("Varaivel observada: ", y , '\n')

#1. visualização dos dados com um gráfico de dispressão
plt.show()