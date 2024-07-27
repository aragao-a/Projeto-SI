import connection as cn
import numpy as np
from pathlib import Path

ROOT_PATH = Path(__file__).parent

s = cn.connect(2037)

def main():
    # lendo o arquivo de resultado.txt
    data = np.loadtxt(ROOT_PATH / 'resultado.txt')
    
    # Parâmetros Q-learning
    alpha = 0.8 # taxa de aprendizado
    gamma = 0.9 # fator de desconto
    epsilon = 0.1 # chance de explorar

    # Variáveis de controle
    # Direções do jogo: Norte = 0, Sul = 1, Leste = 2, Oeste = 3
    # Ações possíveis: left, right e jump
    actions = ['left', 'right', 'jump']

    # Obter o estado inicial
    estado, recompensa = cn.get_state_reward(s, "jump")
    plataforma = int(estado[2:7], 2) # pega do 2º em diante, pois é um binário
    sentido = int(estado[-2:], 2)  # pega os dois últimos índices para o sentido
    print(f'Plataforma: {plataforma}, Sentido: {sentido}, Recompensa: {recompensa}')

    while True:
        
        estado_int = int(estado, 2)
        # Escolher a ação
        if np.random.rand() < epsilon:
            acao = np.random.choice(actions)
        else:
            acao = actions[np.argmax(data[estado_int, :])]
        
       # Salvar o estado atual
        estado_atual = estado_int

        # Executar a ação
        estado, recompensa = cn.get_state_reward(s, acao)
        estado_int = int(estado, 2)
        plataforma = int(estado[2:7], 2)
        sentido = int(estado[-2:], 2)
        print(f'Estado: {estado_int}, Recompensa: {recompensa}, Plataforma: {plataforma}, Sentido: {sentido}')

        # Atualizar a Q-table usando o estado atual e o novo estado
        data[estado_atual, actions.index(acao)] += alpha * (
            recompensa + gamma * np.max(data[estado_int, :]) - data[estado_atual, actions.index(acao)]
        )

        # Salvar a Q-table atualizada de volta no arquivo
        np.savetxt(ROOT_PATH /'resultado.txt', data)

main()
