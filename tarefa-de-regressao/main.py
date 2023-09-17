import numpy as np
import matplotlib.pyplot as plt

#1. visualização dos dados com um gráfico de dispressão
def criarGrafico(tituloEixoY, tituloEixoX, labelDadoPlotado, titulo, X, Y, exibirGrafico):
    if(exibirGrafico):
        plt.scatter(X ,Y , label=labelDadoPlotado)
        plt.xlabel(tituloEixoX)
        plt.ylabel(tituloEixoY)
        plt.title(titulo)
        plt.legend()
        plt.grid(True)
        plt.show() ## -> PLOTAR GRAFICO

TITULOS_EIXOS = {
    "X" :'Temperatura (°C)', 
    "Y" : 'Vendas de Sorvete (unidades)'
}

#3.Assim, defina essa quantidade de rodadas com o valor 1000.
RODADAS_DE_TREINAMENTO = 1000

#1.Extração de dados para visualização dos dados com um gráfico de dispressão
Data = np.loadtxt("base-de-dados/Ice_cream selling data.csv",delimiter=',', skiprows = 1)
temperatura = Data[:, 0]  # Armazena a primeira coluna em coluna1
vendas_sorvete = Data[:, 1]

#2. Definição dos shapes para cada estrutura de dado
N = temperatura.shape[0]
p = 1
X = temperatura.copy().reshape(N,p)
Y = vendas_sorvete.copy().reshape(N, 1)

#4.
MQO_TRADICIONAL = [] #Modelo com intercepitor

MQO_REGULARIZADO = [] #Modelo sem interceptor (MQO regularizado)

MSE_MEDIA = [] #Modelo: Média de valores observáveis

print(X, "temperatura")
print(Y, "vendas")

#1. Plotando grafico
criarGrafico(
    TITULOS_EIXOS["X"], TITULOS_EIXOS["Y"],
    'Dados de Vendas de Sorvete','Relação entre Temperatura e Vendas de Sorvete',
    X, Y, False)


#2. variaveis regressoras sejam armazenadas em uma matriz de dimensão N×p e o mesmo para varaiveis observadas, organizando em um vetor de dimensão N×1
#ESTAMOS INDO NO CAMINHO CERTO
#TEMOS QUE ADICIONAR O INTERCEPTOR NO VETOR X (POR QUE FAZER ISSO)
