import math
import numpy as np
import pandas as pd
import math
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm  # Para progress bar


'''
Resumo do código:

É feita uma grid search em theta1 e theta2 e, para par (theta1, theta2), é realizada toda uma simulação
de execução com saídas antecipadas munidas com um buffer. A taxa de chegada de novos elementos ao buffer é lambda.

As 5 melhores combinações de theta1 e theta2 são exibidas e plotadas.

A função de recompensa segue o paper em construção.

O que resta ser feito:

- Reescrever theta1 e theta2 como variáveis dependentes do tempo.
Isso passa por encontrar funções theta1(t) e theta2(t) tais que: alpha(t, q) = theta1(t)*q + theta2(t).
Recomendo o uso provisorio de funcoes lineares f1(t) = a1*t e f2(t) = a2*t em theta1 e theta2 até encontrarmos aproximações
melhores. As aproximações dependerão do quão sensível ao efeito do tempo de execução o software será.

'''


class ParametrizedUCB:
    def __init__(self, n_arms, theta1, theta2):
        """
        n_arms   : número de braços
        theta1   : seu parâmetro θ1
        theta2   : seu parâmetro θ2
        """
        self.n_arms = n_arms
        self.theta1 = theta1
        self.theta2 = theta2

        # para UCB1
        self.counts = [0] * n_arms         # quantas vezes cada braço foi puxado
        self.values = [0.0] * n_arms       # recompensa média de cada braço
        self.total_counts = 0              # total de pulls realizados

    def select_arm(self, q):
        """
        q : tamanho atual da fila, usado para checar o limiar α_t.
        Retorna o índice do braço a ser puxado.
        Pode lançar ValueError se α_t > 1.
        """
        # 1) checa a restrição global
        alpha = self.theta1 * q + self.theta2
        if alpha > 1.0:
            raise ValueError(f"Alpha_t = {alpha:.4f} > 1 para q = {q}")

        # 2) fase de exploração inicial: puxa cada braço uma vez
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i

        # 3) cálculo dos índices UCB
        ucb_scores = []
        ln_total = math.log(self.total_counts)
        for i in range(self.n_arms):
            bonus = math.sqrt((2 * ln_total) / self.counts[i])
            ucb_scores.append(self.values[i] + bonus)

        # 4) escolhe o braço com maior UCB
        return int(max(range(self.n_arms), key=lambda i: ucb_scores[i]))

    def update(self, chosen_arm, reward):
        """
        chosen_arm : índice do braço que foi puxado
        reward     : recompensa observada
        Atualiza médias incrementais.
        """
        # incrementa contadores
        self.counts[chosen_arm] += 1
        self.total_counts += 1

        # atualização incremental da média
        n = self.counts[chosen_arm]
        current_value = self.values[chosen_arm]
        # nova média: ( (n-1)/n )*old + (1/n)*reward
        self.values[chosen_arm] = ((n - 1) / n) * current_value + (reward / n)


class ConstrainedGridSearch:
    def __init__(self, df, lam, T, n_arms, mu, kappa, q_max=5):
        self.df = df
        self.lam = lam
        self.T = T
        self.n_arms = n_arms
        self.mu = mu
        self.kappa = kappa
        self.q_max = q_max  # Estimativa máxima razoável da fila

    def generate_theta_pairs(self, num_points=20):
        """Gera pares válidos (θ1, θ2) que satisfazem as restrições"""
        valid_pairs = []

        # Gera valores θ1 entre -1/q_max e -0.01
        theta1_min = -1.0/self.q_max
        theta1_vals = np.linspace(theta1_min, -0.01, num_points)

        # Gera θ2 considerando a restrição θ2 ≤ 1 - θ1*q para q=1
        for theta1 in theta1_vals:
            theta2_max = 1 - theta1*1  # Garante 1 ≥ θ1*q + θ2 para q ≥ 1
            theta2_min = 0.1
            if theta2_max < theta2_min:
                continue

            theta2_vals = np.linspace(theta2_min, theta2_max, num_points)
            valid_pairs.extend([(theta1, theta2) for theta2 in theta2_vals])

        return valid_pairs

    def run_search(self, seeds=5):
        """Executa o grid search com restrições dinâmicas"""
        theta_pairs = self.generate_theta_pairs()
        results = []

        for theta1, theta2 in tqdm(theta_pairs, desc="Grid Search Progress"):
            seed_rewards = []
            for seed in range(seeds):
                np.random.seed(seed)
                reward = self.simulate(theta1, theta2)
                seed_rewards.append(reward)

            avg_reward = np.mean(seed_rewards)
            results.append((theta1, theta2, avg_reward))

        return sorted(results, key=lambda x: -x[2])

    def simulate(self, theta1, theta2):
        """Simulação com restrições incorporadas"""
        ucb = ParametrizedUCB(self.n_arms, theta1, theta2)
        queue = []
        rewards = []
        arrivals = np.random.poisson(self.lam, self.T) # modela as chegadas com taxa lambda
        sample_idx = 0
        N = len(self.df)

        for t in range(self.T):
            # Atualiza fila
            new_arrivals = arrivals[t]
            queue.extend(range(sample_idx, min(sample_idx + new_arrivals, N)))
            sample_idx = min(sample_idx + new_arrivals, N)

            if not queue:
                continue

            # Seleção de braço com verificação de restrições
            q = len(queue)
            try:
                arm = ucb.select_arm(q)
            except ValueError as e:
                print(f"Erro com θ1={theta1}, θ2={theta2}, q={q}: {str(e)}")
                return -np.inf

            # Processa amostra
            idx = queue.pop(0)
            CI = self.df.at[idx, 'conf_branch_1']
            CL = self.df.at[idx, 'conf_branch_2']
            o_q = self.mu * q - self.kappa
            r = (max(CL - CI, 0) - o_q) if CI < (theta1 * q + theta2) else 0 # definicao da recompensa segundo o artigo

            ucb.update(arm, r)
            rewards.append(r)

        return np.mean(rewards) if rewards else 0.0

def plot_constraints(q_max, best_params):
    plt.figure(figsize=(10, 6))

    # Região viável
    theta1 = np.linspace(-1/q_max, -0.01, 100)
    theta2 = 1 - theta1*1  # Limite superior

    plt.fill_between(theta1, theta2, 2.0, color='red', alpha=0.1, label='Região Inválida')
    plt.fill_between(theta1, 0.1, theta2, color='green', alpha=0.1, label='Região Viável')

    # Melhores parâmetros
    best_theta1 = [p[0] for p in best_params[:5]]
    best_theta2 = [p[1] for p in best_params[:5]]
    plt.scatter(best_theta1, best_theta2, color='blue', s=100, edgecolor='black', label='Top 5 Parâmetros')

    plt.title('Espaço de Parâmetros com Restrições', fontsize=14)
    plt.xlabel('θ1', fontsize=12)
    plt.ylabel('θ2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

# no jupyer notebook:
if __name__ == '__main__':
    # Configurações
    lam, T, n_arms = 2.0, 2000, 5
    mu, kappa = 0.05, 0.1
    q_max = 5  # estimativa de tamanho maximo da fila

    # Executar grid search
    searcher = ConstrainedGridSearch(df, lam, T, n_arms, mu, kappa, q_max) # demora (bastante), mas podemos usar um hill climbing ou algo assim pra otimizar e encontrar respostas ok
    best_params = searcher.run_search(seeds=5)

    # Resultados
    print("\nTop 5 parâmetros:")
    for theta1, theta2, reward in best_params[:5]:
        print(f"θ1={theta1:.4f}, θ2={theta2:.4f} → Recompensa: {reward:.4f}")

    # Plotagem com restrições
    plot_constraints(q_max, best_params)
