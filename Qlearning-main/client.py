#Aqui vocês irão colocar seu algoritmo de aprendizado

import connection as cn

s = cn.connect(2037)

estado, recompensa = cn.get_state_reward(s, "jump")