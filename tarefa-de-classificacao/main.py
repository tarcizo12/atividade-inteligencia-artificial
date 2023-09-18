import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


    #Modelo MQO tradicionao - com interceptor
    sizeX_treino = X_treino.shape[0]
    interceptorTreino = np.ones((sizeX_treino , 1)) 
    X_treino = np.concatenate((interceptorTreino , X_treino),axis=1)
    MODELO_MQO_TRADICIONAL = np.linalg.pinv(X_treino.T@X_treino)@X_treino.T@Y_treino

#print(X.shape)