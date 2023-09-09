import numpy as np
import matplotlib.pyplot as plt

# Dados
Data = np.loadtxt("base-de-dados/Ice_cream selling data.csv",delimiter=',', skiprows = 1)
temperatura = Data[:, 0]  # Armazena a primeira coluna em coluna1
vendas_sorvete = Data[:, 1]

print(temperatura)


 ##Criar o gráfico de dispersão
plt.scatter(temperatura, vendas_sorvete, label='Dados de Vendas de Sorvete')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vendas de Sorvete (unidades)')
plt.title('Relação entre Temperatura e Vendas de Sorvete')
plt.legend()
plt.grid(True)

X = np.vstack((temperatura)).T
y = np.array(vendas_sorvete).reshape(-1, 1)

## Mostrar o gráfico
plt.show()