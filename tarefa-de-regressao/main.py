import numpy as np
import matplotlib.pyplot as plt

#1. visualização dos dados com um gráfico de dispressão
def criarGraficoDispressao(tituloEixoY, tituloEixoX, labelDadoPlotado, titulo, X, Y, exibirGrafico):
    if(exibirGrafico):
        plt.scatter(X ,Y , label=labelDadoPlotado)
        plt.xlabel(tituloEixoX)
        plt.ylabel(tituloEixoY)
        plt.title(titulo)
        plt.legend()
        plt.grid(True)
        plt.show() ## -> PLOTAR GRAFICO

def criarTabela(valores, labelX, labelY,labelZ, title):
    labels = [labelX, labelY, labelZ]
    valores = [valores[0], valores[1], valores[2]]

    fig, ax = plt.subplots(figsize=(4, 4))
    
    #Criar o gráfico de barras
    ax.bar(labels, valores, color=['blue', 'green', 'red'])

    #Linha tracejada no valor máximo
    for i, valor in enumerate(valores):
        ax.axhline(valor, color='black', linestyle='--', label=f'Limite {labels[i]}: {valor}')
    
    
    plt.title(title)

def definirAlphaMinimo(rodadasDeTreino, X, Y):
    # Gere N valores no intervalo 0 < λ ≤ 1
    valoresParaAlpha =  np.arange(0.001, 1.001, 0.001) 
    alphaMinimo = -1
    minValue = 100000

    for alphaAtual in valoresParaAlpha:
        mediaEQM = []

        for rodada in range(rodadasDeTreino):
            indexRandom = np.random.permutation(N)
            indexOfOitentaPorCento = int(N*.8)

            #Embaralhar dados
            X_embaralhado = X[indexRandom,:]
            Y_embaralhado = Y[indexRandom,:]

            #6. Amostra para treino e teste 
            X_treino = X_embaralhado[0: indexOfOitentaPorCento,:] #Ir de Zero até o index 80% total (no caso é 39)
            Y_treino = Y_embaralhado[0: indexOfOitentaPorCento,:]
            X_teste =  X_embaralhado[indexOfOitentaPorCento: N,:] #Ir do ultimo index que representa os 80% até o fim
            Y_teste =  Y_embaralhado[indexOfOitentaPorCento: N,:]

  
            modelo_mqo_regularizado = np.linalg.inv((X_treino.T @ X_treino) + alphaAtual * np.identity((X_treino.T @ X_treino).shape[0]))@ X_treino.T @ Y_treino
            mediaEQM.append(modelo_mqo_regularizado[0][0])
        
        mediaAlphaAtual = np.mean(mediaEQM)

        if(mediaAlphaAtual < minValue):
            minValue = mediaAlphaAtual
            alphaMinimo = alphaAtual

    print(alphaMinimo)
    return alphaMinimo

TITULOS_EIXOS = {
    "X" :'Temperatura (°C)', 
    "Y" : 'Vendas de Sorvete (unidades)'
}

#3.Assim, defina essa quantidade de rodadas com o valor 1000.
RODADAS_DE_TREINAMENTO = 100

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

MQO_REGULARIZADO = [] #Modelo regularizado com lambda

MSE_MEDIA = [] #Modelo: Média de valores observáveis

alphaMinimo = definirAlphaMinimo(RODADAS_DE_TREINAMENTO, X, Y)

for r in range(RODADAS_DE_TREINAMENTO):
    indexRandom = np.random.permutation(N)
    indexOfOitentaPorCento = int(N*.8)

    #6. Embaralhar dados
    X_embaralhado = X[indexRandom,:]
    Y_embaralhado = Y[indexRandom,:]

    #6. Amostra para treino e teste 
    X_treino = X_embaralhado[0: indexOfOitentaPorCento,:] #Ir de Zero até o index 80% total (no caso é 39)
    Y_treino = Y_embaralhado[0: indexOfOitentaPorCento,:]
    X_teste =  X_embaralhado[indexOfOitentaPorCento: N,:] #Ir do ultimo index que representa os 80% até o fim
    Y_teste =  Y_embaralhado[indexOfOitentaPorCento: N,:]

    #Media dos valores de Y
    mediaDosValoresDeTeste_Y = np.mean(Y_treino)
    mediaDosValoresDeTeste_Y = np.array([
        [mediaDosValoresDeTeste_Y],
        [0],
    ])

    #Modelo MQO tradicionao - com interceptor
    sizeX_treino = X_treino.shape[0]
    interceptorTreino = np.ones((sizeX_treino , 1)) 
    X_treino = np.concatenate((interceptorTreino , X_treino),axis=1)
    MODELO_MQO_TRADICIONAL = np.linalg.pinv(X_treino.T@X_treino)@X_treino.T@Y_treino

    #Modelo MQO regularizado
    MODELO_MQO_REGULARIZADO = np.linalg.inv((X_treino.T @ X_treino) + alphaMinimo * np.identity((X_treino.T @ X_treino).shape[0]))@ X_treino.T @ Y_treino
    

    #Calcular médias EQM
    interceptorTeste = np.ones((X_teste.shape[0],1))
    X_teste = np.concatenate((interceptorTeste,X_teste),axis=1)

    predicaoMedia = X_teste@mediaDosValoresDeTeste_Y
    predicaoModeloTradicional = X_teste@MODELO_MQO_TRADICIONAL
    peridocaoModeloRegularizado = X_teste@MODELO_MQO_REGULARIZADO
    
    #Determinnar distancias
    MSE_MEDIA.append(np.mean((Y_teste-predicaoMedia)**2))
    MQO_TRADICIONAL.append(np.mean((Y_teste-predicaoModeloTradicional)**2))
    MQO_REGULARIZADO.append(np.mean((Y_teste-peridocaoModeloRegularizado)**2))

#Médias de EQM
media_MVO = np.mean(MSE_MEDIA)
media_MQO_TRADICIONAL = np.mean(MQO_TRADICIONAL)
media_MQO_REGULARIZADO = np.mean(MQO_REGULARIZADO)

#DESVIO PADRÃO DE EQM
desvioPadrao_MVO = np.std(MSE_MEDIA)
desvioPadrao_MQO_TRADICIONAL = np.std(MQO_TRADICIONAL)
desvioPadrao_MQO_REGULARIZADO = np.std(MQO_REGULARIZADO)

#Maiores e menores valores
menorValor_MVO = np.amin(MSE_MEDIA)
maiorValor_MVO = np.amax(MSE_MEDIA)
menorValor_MQO_TRADICIONAL = np.amin(MQO_TRADICIONAL)
maiorValor_MQO_TRADICIONAL = np.amax(MQO_TRADICIONAL)
menorValor_MQO_REGULARIZADO = np.amin(MQO_REGULARIZADO)
maiorValor_MQO_REGULARIZADO = np.amax(MQO_REGULARIZADO)

# print(media_MVO, "Media para o MVO \n")
# print(media_MQO_TRADICIONAL, "Media para o MQO TRADICIONAL \n")
# print(desvioPadrao_MVO , "Desvio Padrao para o MVO \n")
# print(desvioPadrao_MQO_TRADICIONAL, "Desvio Padrao para o MQO TRADICIONAL \n")
# print(menorValor_MVO, maiorValor_MVO, "Menor e maior valor para MVO \n")
# print(menorValor_MQO_TRADICIONAL, maiorValor_MQO_TRADICIONAL, "Menor e maior valor para MQO tradicional \n")


iniciarTabelasInformativas = False
iniciarGraficoDispresao = False
#1. Plotando grafico
criarGraficoDispressao(
    TITULOS_EIXOS["X"], TITULOS_EIXOS["Y"],
    'Dados de Vendas de Sorvete','Relação entre Temperatura e Vendas de Sorvete',
    X, Y, iniciarGraficoDispresao)

if(iniciarTabelasInformativas):
    criarTabela([media_MVO, media_MQO_TRADICIONAL, media_MQO_REGULARIZADO], 'Média valores Observaveis', 'MQO tradicional','MQO regularizado','Média de EQM para os modelos')
    criarTabela([desvioPadrao_MVO, desvioPadrao_MQO_TRADICIONAL, desvioPadrao_MQO_REGULARIZADO], 'Média valores Observaveis', 'MQO tradicional','MQO regularizado','Desvio padrão')
    criarTabela([menorValor_MVO, menorValor_MQO_TRADICIONAL, menorValor_MQO_REGULARIZADO], 'Média valores Observaveis', 'MQO tradicional','MQO regularizado','Menores valores de EQM')
    criarTabela([maiorValor_MVO, maiorValor_MQO_TRADICIONAL, maiorValor_MQO_REGULARIZADO], 'Média valores Observaveis', 'MQO tradicional','MQO regularizado','Maiores valores de EQM')
    
    # Exibir o gráfico
    plt.show()

#1. Plotando grafic


#2. variaveis regressoras sejam armazenadas em uma matriz de dimensão N×p e o mesmo para varaiveis observadas, organizando em um vetor de dimensão N×1
#ESTAMOS INDO NO CAMINHO CERTO
#TEMOS QUE ADICIONAR O INTERCEPTOR NO VETOR X (POR QUE FAZER ISSO)
