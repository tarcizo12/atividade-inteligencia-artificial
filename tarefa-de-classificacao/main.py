import pandas as pd
import matplotlib.pyplot as plt

# Vamos supor que o arquivo está no mesmo diretório do script e se chama 'dados.csv'
df = pd.read_csv('./base-de-dados/EMG.csv', header=None,
                 names=['Sensor1', 'Sensor2'])

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
