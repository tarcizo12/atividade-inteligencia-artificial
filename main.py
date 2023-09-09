import numpy as np
import matplotlib.pyplot as plt

# Dados
temperatura = np.array([-4.662262677220208, -4.316559446725467, -4.213984764590729, -3.9496610890515707, -3.578553716228682, -3.455711698065576, -3.1084401208909964, -3.0813033243034563, -2.672460827006454])
vendas_sorvete = np.array([41.84298632027783, 34.661119537360234, 39.38300087682567, 37.53984488250128, 32.28453118789761, 30.00113847641735, 22.635401277012628, 25.36502221208036, 19.226970048254086])

# Criar o gráfico de dispersão
plt.scatter(temperatura, vendas_sorvete, label='Dados de Vendas de Sorvete')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vendas de Sorvete (unidades)')
plt.title('Relação entre Temperatura e Vendas de Sorvete')
plt.legend()
plt.grid(True)

# Mostrar o gráfico
plt.show()