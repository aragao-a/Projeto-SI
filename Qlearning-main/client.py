import connection as cn
import numpy as np

s = cn.connect(2037)

estado, recompensa = cn.get_state_reward(s, "jump")
plataforma = estado[2:7] # pega do 2º em diante, pois é um binário
sentido = estado[-2:]  # pega os dois últimos índices para o sentido

