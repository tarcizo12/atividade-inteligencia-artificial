import numpy as np
import matplotlib.pyplot as plt

#1. visualização dos dados com um gráfico de dispressão
def criarGrafico(tituloEixoY, tituloEixoX, labelPrincipal, titulo, X, Y):
    plt.scatter(X ,Y , label=labelPrincipal)
    plt.xlabel(tituloEixoX)
    plt.ylabel(tituloEixoY)
    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    plt.show() ## -> PLOTAR GRAFICO

#1.Extração de dados para visualização dos dados com um gráfico de dispressão 
Data = np.loadtxt("base-de-dados/Ice_cream selling data.csv",delimiter=',', skiprows = 1)
temperatura = Data[:, 0]  # Armazena a primeira coluna em coluna1
vendas_sorvete = Data[:, 1]


criarGrafico(
    'Temperatura (°C)','Vendas de Sorvete (unidades)',
    'Dados de Vendas de Sorvete','Relação entre Temperatura e Vendas de Sorvete',
    temperatura, vendas_sorvete)



#2. variaveis regressoras sejam armazenadas em uma matriz de dimensão N×p
##  O mesmo para o vetor de varaiveis observadas, organizando em um vetor de dimensão N×1
X = np.vstack((temperatura)).T
y = np.array(vendas_sorvete).reshape(-1, 1)