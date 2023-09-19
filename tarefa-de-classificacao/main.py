import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Vamos supor que o arquivo está no mesmo diretório do script e se chama 'dados.csv'
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

            descriminante_predicao = np.argmax(Y_predicao, axis=1)
            descriminante_teste = np.argmax(Y_teste, axis=1)
            acuaria_mqo_regularizado = accuracy_score(descriminante_predicao, descriminante_teste)

            valoresDeAcuracias.append(acuaria_mqo_regularizado)
        
        if(np.mean(valoresDeAcuracias) > maxValue):
            maxValue = np.mean(valoresDeAcuracias)
            alphaMaximoProcurado = alphaAtual

    return alphaMaximoProcurado

def determinarAcuracia(X_Teste, Y_teste,MODELO, label):
    Y_predicao = X_Teste @ MODELO

    descriminante_predicao = np.argmax(Y_predicao, axis=1)
    descriminante_teste = np.argmax(Y_teste, axis=1)
    acuracia_modelo = accuracy_score(descriminante_predicao, descriminante_teste)

    if(label != ""):
        print("Modelo: " , label ,", Acurácia: " , acuracia_modelo , "\n")

    return acuracia_modelo 

def dmc(X_treino, Y_treino, X_teste):
    classes = np.unique(Y_treino)
    previsoes = []
    i = 0 
    for amostra_teste in X_teste:
        print(i)
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
if(False):
    for i, expressao in enumerate(expressoes):
        dados_expressao = df[df['Expressao'] == expressao]
        plt.scatter(dados_expressao['Sensor1'], dados_expressao['Sensor2'],
                    label=expressao, color=[cores_por_expressao[i]])

    plt.xlabel('Sensor1')
    plt.ylabel('Sensor2')
    plt.title('Gráfico de Dispersão dos Sensores por Expressão Facial')
    plt.legend()
    plt.show()



# 2
RODADAS_DE_TREINAMENTO = 1

##Modelos para implementação
MQO_TRADICIONAL = [] #Modelo com intercepitor
MQO_REGULARIZADO = []


melhorAlpha = encontrarAlpha(RODADAS_DE_TREINAMENTO, X , Y)

acuracia_MQO_TRADICIONAL_registros = []
acuracia_MQO_REGULARIZADO_registros = []


interceptorTreino = np.ones((X.shape[0] , 1)) 
X = np.concatenate((interceptorTreino , X),axis=1)


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

    modelo_dmc = dmc(X_treino, np.argmax(Y_treino, axis=1), X_teste)
    descriminante_y_teste = np.argmax(Y_teste,axis=1)
    acuracia_dmc =  accuracy_score(modelo_dmc, descriminante_y_teste)

    print(acuracia_dmc)


    