import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import statistics


##Exibir graficos
exibirInformacoesDeMedia = False
plotarGraficoDispresao =  False
exibirInformacoesDeDesvioPadrao = False 
exibirInformacoesDeMaioresEMenoresValores = False
exibirModaDeCadaModelo = False
iniciarTabelasInformativas =  True

# 2. Definicao de rodadas
RODADAS_DE_TREINAMENTO = 100

##Funcoes
def encontrarAlpha(rodadasDeTreino, X, Y):
    # Gere N valores no intervalo 0 < λ ≤ 1
    valoresParaAlpha =  np.arange(0.01, 1.01, 0.01) 
    alphaMaximoProcurado = 1
    maxValue = -1

    for alphaAtual in valoresParaAlpha:
        valoresDeAcuracias = []
        print("Alfa atual: ", alphaAtual)
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

            Y_predicao = X_teste @ modelo_mqo_regularizado

            predicoes_classe = np.argmax(Y_predicao, axis=1)
            classes_verdadeiras = np.argmax(Y_teste, axis=1)
            acuaria_mqo_regularizado = accuracy_score(predicoes_classe, classes_verdadeiras)

            valoresDeAcuracias.append(acuaria_mqo_regularizado)
        
        if(np.mean(valoresDeAcuracias) > maxValue):
            maxValue = np.mean(valoresDeAcuracias)
            alphaMaximoProcurado = alphaAtual

    return alphaMaximoProcurado

def encontrarValorDeK(rodadasDeTreino, X, Y):
    # Gere N valores no intervalo 0 < λ ≤ 1
    valoresParaK =  list(range(1, 21))
    kProcurado = -1
    maxValue = -1

    for kAtual in valoresParaK:
        valoresDeAcuracias = []
        print("K atual: ", kAtual)
        
        if(kAtual % 2 != 0):
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
    
                k = kAtual
                MODELO_KNN = KNeighborsClassifier(n_neighbors= k)
                MODELO_KNN.fit(X_treino, Y_treino)
                Y_predicao_knn = MODELO_KNN.predict(X_teste)
                acuracia_knn = accuracy_score(Y_teste, Y_predicao_knn)
                valoresDeAcuracias.append(acuracia_knn)
        
            if(np.mean(valoresDeAcuracias) > maxValue):
                maxValue = np.mean(valoresDeAcuracias)
                kProcurado = kAtual

    return kProcurado

def determinarAcuracia(X_Teste, Y_teste,MODELO, label):
    Y_predicao = X_Teste @ MODELO

    predicoes_classe = np.argmax(Y_predicao, axis=1)
    classes_verdadeiras = np.argmax(Y_teste, axis=1)
    acuracia_modelo = accuracy_score(predicoes_classe, classes_verdadeiras)

    if(label != ""):
        print("Modelo: " , label ,", Acurácia: " , acuracia_modelo , "\n")

    return acuracia_modelo 

def dmc(X_treino, Y_treino, X_teste):
    classes = np.unique(Y_treino)
    previsoes = []
    i = 0 
    for amostra_teste in X_teste:
        print("Amostra de dmc atual: ", i)
        distancias = []
        i = i + 1
        for classe in classes:
            centroides_classe = X_treino[Y_treino == classe]
            centroide = np.mean(centroides_classe, axis=0)
            distancia = np.linalg.norm(amostra_teste - centroide)
            distancias.append(distancia)
        
        classe_mais_proxima = classes[np.argmin(distancias)]
        previsoes.append(classe_mais_proxima)

    return np.array(previsoes)

def printInformacao(label, valor, modelo):
    print(f'\n{label} {modelo} {valor}%')

def criarTabela(valores, labelX, labelY,labelZ,labelW, title):
    labels = [labelX, labelY, labelZ, labelW]
    valores = [valores[0], valores[1], valores[2], valores[3]]

    fig, ax = plt.subplots(figsize=(4, 4))

    # Criar o gráfico de barras
    ax.bar(labels, valores, color=['blue', 'green', 'red', 'purple'])

    # Linha tracejada no valor máximo
    for i, valor in enumerate(valores):
        ax.axhline(valor, color='black', linestyle='--', label=f'Limite {labels[i]}: {valor}')

    plt.title(title)

    num_ticks = 15  
    ax.yaxis.set_major_locator(ticker.LinearLocator(num_ticks))

##Extraindo informações
df = pd.read_csv('./base-de-dados/EMG.csv', header=None,
                 names=['Sensor1', 'Sensor2'])

#Definição para as variaveis X e Y
X = df.values
N,p = X.shape

neutro = np.tile(np.array([[1,-1,-1,-1,-1]]),(1000,1)) 
sorrindo = np.tile(np.array([[-1,1,-1,-1,-1]]),(1000,1)) 
aberto = np.tile(np.array([[-1,-1,1,-1,-1]]),(1000,1)) 
surpreso = np.tile(np.array([[-1,-1,-1,1,-1]]),(1000,1)) 
rabugento = np.tile(np.array([[-1,-1,-1,-1,1]]),(1000,1)) 
Y = np.tile(np.concatenate((neutro,sorrindo,aberto,surpreso,rabugento)),(10,1))

# Identificar a expressão correspondente a cada bloco de 10.000 observações
expressoes = ['Neutro', 'Sorrindo', 'Aberto', 'Surpreso', 'Rabugento']
expressoes_repetidas = [expressao for expressao in expressoes for _ in range(10000)]

# Adicionar a coluna de expressões ao DataFrame
df['Expressao'] = expressoes_repetidas

# Criar um mapa de cores para as expressões
cores = plt.cm.viridis.colors

# Selecionar cores diferentes para cada expressão
cores_por_expressao = [
    cores[i * len(cores) // len(expressoes)] for i in range(len(expressoes))]


# Plotar cada expressão facial separadamente para adicionar legenda
if(plotarGraficoDispresao):
    for i, expressao in enumerate(expressoes):
        dados_expressao = df[df['Expressao'] == expressao]
        plt.scatter(dados_expressao['Sensor1'], dados_expressao['Sensor2'],
                    label=expressao, color=[cores_por_expressao[i]])

    plt.xlabel('Sensor1')
    plt.ylabel('Sensor2')
    plt.title('Gráfico de Dispersão dos Sensores por Expressão Facial')
    plt.legend()
    plt.show()


#Precisão
acuracia_MQO_TRADICIONAL_registros = []
acuracia_MQO_REGULARIZADO_registros = []
acuracia_DMC_registros = []
acuracia_KNN_registros = []


interceptorTreino = np.ones((X.shape[0] , 1)) 
X = np.concatenate((interceptorTreino , X),axis=1)

melhorAlpha = encontrarAlpha(RODADAS_DE_TREINAMENTO, X , Y)
melhorK = encontrarValorDeK(RODADAS_DE_TREINAMENTO, X, Y)
print(melhorK)

for rodada in range(RODADAS_DE_TREINAMENTO):
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

    #Modelo MQO regularizado 
    MODELO_MQO_REGULARIZADO = np.linalg.inv((X_treino.T @ X_treino) + melhorAlpha * np.identity((X_treino.T @ X_treino).shape[0]))@ X_treino.T @ Y_treino
    acuracia_mqo_regularizado = determinarAcuracia(X_teste, Y_teste, MODELO_MQO_REGULARIZADO, "")
    acuracia_MQO_REGULARIZADO_registros.append(acuracia_mqo_regularizado)


    #Modelo MQO tradicional - com interceptor
    MODELO_MQO_TRADICIONAL = np.linalg.pinv(X_treino.T@X_treino)@X_treino.T@Y_treino
    acuracia_mqo_tradicional = determinarAcuracia(X_teste, Y_teste, MODELO_MQO_TRADICIONAL, "")
    acuracia_MQO_TRADICIONAL_registros.append(acuracia_mqo_tradicional)


    #Dmc
    MODELO_DMC = dmc(X_treino, np.argmax(Y_treino, axis=1), X_teste)
    predicao_Y_teste = np.argmax(Y_teste,axis=1)
    acuracia_dmc =  accuracy_score(MODELO_DMC, predicao_Y_teste)
    acuracia_DMC_registros.append(acuracia_dmc)

    #k-nn
    k = melhorK
    MODELO_KNN = KNeighborsClassifier(n_neighbors= k)
    MODELO_KNN.fit(X_treino, Y_treino)
    Y_predicao_knn = MODELO_KNN.predict(X_teste)
    acuracia_knn = accuracy_score(Y_teste, Y_predicao_knn)
    acuracia_KNN_registros.append(acuracia_knn)


media_dmc = np.mean(acuracia_DMC_registros)*100
media_knn = np.mean(acuracia_KNN_registros)*100
media_mqo_tradicional = np.mean(acuracia_MQO_TRADICIONAL_registros)*100
media_mqo_regularizado = np.mean(acuracia_MQO_REGULARIZADO_registros)*100

desvioPadrao_dmc = np.std(acuracia_DMC_registros)
desvioPadrao_knn =  np.std(acuracia_KNN_registros)
desvioPadrao_mqo_tradicional = np.std(acuracia_MQO_TRADICIONAL_registros)
desvioPadrao_mqo_regularizado = np.std(acuracia_MQO_REGULARIZADO_registros)

#Menores Valores
menorValor_dmc = np.amin(acuracia_DMC_registros)*100
menorValor_knn = np.amin(acuracia_KNN_registros)*100
menorValor_mqo_tradicional = np.amin(acuracia_MQO_TRADICIONAL_registros)*100
menorValor_mqo_regularizado = np.amin(acuracia_MQO_REGULARIZADO_registros)*100

#MaioresValores
maiorValor_dmc = np.amax(acuracia_DMC_registros)*100
maiorValor_knn = np.amax(acuracia_KNN_registros)*100
maiorValor_mqo_tradicional = np.amax(acuracia_MQO_TRADICIONAL_registros)*100
maiorValor_mqo_regularizado = np.amax(acuracia_MQO_REGULARIZADO_registros)*100

#Moda
moda_dmc = statistics.mode(acuracia_DMC_registros)*100
moda_knn = statistics.mode(acuracia_KNN_registros)*100
moda_mqo_tradicional = statistics.mode(acuracia_MQO_TRADICIONAL_registros)*100
moda_mqo_regularizado = statistics.mode(acuracia_MQO_REGULARIZADO_registros)*100


if(exibirInformacoesDeMedia):
    printInformacao("Media:", media_dmc, "DCM")
    printInformacao("Media:", media_knn,"Knn")
    printInformacao("Media:", media_mqo_tradicional, "MQO Tradicional")
    printInformacao("Media:",media_mqo_regularizado,"MQO Regularizado")


if(exibirInformacoesDeDesvioPadrao):
    printInformacao("Desvio:", desvioPadrao_dmc, "DCM")
    printInformacao("Desvio:", desvioPadrao_knn,"Knn")
    printInformacao("Desvio:", desvioPadrao_mqo_tradicional, "MQO Tradicional")
    printInformacao("Desvio:",desvioPadrao_mqo_regularizado,"MQO Regularizado")


if(exibirInformacoesDeMaioresEMenoresValores):
    printInformacao("Maior valor:", maiorValor_dmc, "DCM")
    printInformacao("Maior valor:", desvioPadrao_knn,"Knn")
    printInformacao("Maior valor:", maiorValor_knn, "MQO Tradicional")
    printInformacao("Maior valor:", maiorValor_mqo_regularizado,"MQO Regularizado")
    
    printInformacao("Menor Valor:", menorValor_dmc, "MQO Tradicional")
    printInformacao("Menor Valor:",menorValor_knn,"MQO Regularizado")
    printInformacao("Menor Valor:",menorValor_mqo_tradicional,"MQO Regularizado")
    printInformacao("Menor Valor:",menorValor_mqo_regularizado,"MQO Regularizado")

if(exibirModaDeCadaModelo):
    printInformacao("Moda:", moda_dmc, "DCM")
    printInformacao("Moda:", moda_knn,"Knn")
    printInformacao("Moda:", moda_mqo_tradicional, "MQO Tradicional")
    printInformacao("Moda:", moda_mqo_regularizado,"MQO Regularizado")



##Plotar Grafico de tabelas
if(iniciarTabelasInformativas):
    criarTabela([media_dmc, media_knn, media_mqo_tradicional, media_mqo_regularizado], 
                'DMC ', 'K-NN','MQO tradicional','MQO regularizado','Média de acurácia para os modelos')
    criarTabela([desvioPadrao_dmc, desvioPadrao_knn, desvioPadrao_mqo_tradicional, desvioPadrao_mqo_regularizado], 
                'DMC ', 'K-NN','MQO tradicional','MQO regularizado','Desvio padrão de acurácia para os modelos')
    criarTabela([menorValor_dmc, menorValor_knn, menorValor_mqo_tradicional, menorValor_mqo_regularizado],
                'DMC ', 'K-NN','MQO tradicional','MQO regularizado','Menores valores de acurácia para os modelos')
    criarTabela([maiorValor_dmc, maiorValor_knn, maiorValor_mqo_tradicional, maiorValor_mqo_regularizado], 
                'DMC ', 'K-NN','MQO tradicional','MQO regularizado','Maiores valores de acurácia para os modelos')
    criarTabela([moda_dmc, moda_knn, moda_mqo_tradicional, moda_mqo_regularizado], 
                'DMC ', 'K-NN','MQO tradicional','MQO regularizado','Moda de acurácia para os modelos')
    
    # Exibir o gráfico
    plt.show()
    

    