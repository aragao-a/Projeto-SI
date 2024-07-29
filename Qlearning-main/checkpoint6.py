import connection as cn
import socket
import numpy as np

def main():
    
    # Inicializa a conexão
    c = cn.connect(2037)
    
    # Verifica se conectou com sucesso
    if isinstance(c, socket.socket):
        
        print("Conectado com sucesso")
        
        # Carrega a tabela Q
        try:
            Q = np.loadtxt('resultado.txt')
        except:
            Q = np.zeros([24 * 4, 3])  # Se der erro inicia a tabela Q com zeros      
        
        # Parâmetros do Q-Learning
        alpha = 0.7 # taxa de aprendizado(0 = não aprende nada, 1 = aprende 100%)
        gamma = 0.95 # fator de desconto(busca de recompensas)
        epsilon = 0.1 # seleção de ação
        
        # Variáveis para o jogo
        # Direções possíveis Norte = 0, Leste = 1, Sul = 2,Oeste = 3
        # Ações possíveis: 0 = left, 1 = right, 2 = jump
        acoesPossiveis = ["left", "right", "jump"]        
        jogo_terminou = False
         
        # Pega o estado e a recompensa iniciais
        estado, recompensa = cn.get_state_reward(c, "")
        plataforma = int(estado[2:7], 2)
        direcao = int(estado[-2:], 2)
        print("\nEstado: "+estado+"\nPlataforma: " + str(plataforma)+"\nDireção: " + str(direcao)+"\nRecompensa: "+str(recompensa)+"\n")
        # Loop do jogo
        while True:       

            # Converte o estado em um número inteiro
            estado_int = int(estado, 2) 

            # Escolhe uma ação
            n = np.random.uniform(0, 1)
            if n < epsilon:
                acao = np.random.choice([0, 1, 2])
            else:
                acao = np.argmax(Q[estado_int])
                            
            # Executa ação e recebe estado e recompensa
            novo_estado, recompensa = cn.get_state_reward(c, acoesPossiveis[acao])
            novo_estado_int = int(novo_estado, 2)            
            plataforma = int(novo_estado[2:7], 2)
            direcao = int(novo_estado[-2:], 2)
            print("Ação efetuada: " + acoesPossiveis[acao] + "\nNovo estado: "+novo_estado+"\nPlataforma: " + str(plataforma)+"\nDireção: " + str(direcao)+"\nRecompensa: "+str(recompensa)+"\n")
             
            # Atualiza a tabela Q
            Q[estado_int, acao] = (1 - alpha) * Q[estado_int, acao] + alpha * (recompensa + gamma * np.max(Q[novo_estado_int]))               
                        
            # Atualiza o estado atual
            estado = novo_estado
            
            # Atualiza a tabela Q
            np.savetxt('resultado.txt', Q)
    else:
        print("Falha na conexão")
main()