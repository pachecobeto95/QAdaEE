import config, ee_dnns, sys, utils
import argparse, logging, os, torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd


def compute_backlog_threshold(vector):
    try:
        # Encontrar o menor índice de 'e' e adicionar 1
        return vector.index('e') + 1
    except ValueError:
        # Se 'e' não estiver no vetor, retorna 0
        return 0

def multi_operator_value_iteration(B, mu_a, mu_b, p_a, p_b, gamma, lambda_rate, max_iterations=1000, tolerance=1e-6):

	# Initialize value functions
	U = np.zeros(B)
	U_a_A = np.zeros(B + 1)  # Arrival state values for "a"
	U_b_A = np.zeros(B + 1)  # Arrival state values for "b"
	U_a_D = np.zeros(B)      # Departure state values for "a"
	U_b_D = np.zeros(B)      # Departure state values for "b"

	# Rewards
	c_a = (1 - p_a) * mu_a / (gamma + mu_a)
	c_b = (1 - p_b) * mu_b / (gamma + mu_b)

	# Precompute factors
	delta_a = 1 / (mu_a + lambda_rate + gamma)
	delta_b = 1 / (mu_b + lambda_rate + gamma)
	beta_a = 1 / (mu_a + gamma)
	beta_b = 1 / (mu_b + gamma)
	delta_bar = 1 / (lambda_rate + gamma)

	for iteration in range(max_iterations):
		U_prev = U.copy()

		# Update arrival values
		for n in range(B):
			U_a_A[n] = mu_a * delta_a * U[n] + lambda_rate * delta_a * U_a_A[n + 1]
			U_b_A[n] = mu_b * delta_b * U[n] + lambda_rate * delta_b * U_b_A[n + 1]

		# Boundary condition for arrival states
		U_a_A[B] = mu_a * delta_a * U[B - 1] + lambda_rate * delta_a * U_a_A[B]
		U_b_A[B] = mu_b * delta_b * U[B - 1] + lambda_rate * delta_b * U_b_A[B]

		# Update departure values
		for n in range(1, B):
			U_a_D[n] = mu_a * delta_a * U[n - 1] + lambda_rate * delta_a * U_a_A[n] + c_a
			U_b_D[n] = mu_b * delta_b * U[n - 1] + lambda_rate * delta_b * U_b_A[n] + c_b

		# Update overall value function
		for n in range(1, B):
			U[n] = max(U_a_D[n], U_b_D[n])

		# Boundary condition for empty buffer
		U[0] = lambda_rate * delta_bar * max(U_a_A[0], U_b_A[0])

		# Check convergence
		if np.max(np.abs(U - U_prev)) < tolerance:
			break

	# Generate decision vector
	decision_vector = ['e' if U_a_D[n] >= U_b_D[n] else 'n' for n in range(B)]
	decision_vector[0] = 'e' if U_a_A[0] >= U_b_A[0] else 'n'


	return U, U_a_A, U_b_A, U_a_D, U_b_D, decision_vector, compute_backlog_threshold(decision_vector)


def save_QAdaEE_results(U, U_a_A, U_b_A, U_a_D, U_b_D, decision_vector, tau, class_name, threshold, lr, B, 
	P_full_buffer, P_class_E, P_class_N, long_term_reward, throughput, misclassification_rate, drop_rate, results_path):

	results_dict = {}

	for i in range(1, B + 1):
		results_dict.update({"U_%s"%(i): [U[i-1]], "U_a_A_%s"%(i): [U_a_A[i-1]], "U_a_D_%s"%(i): [U_a_D[i-1]],
			"U_b_D_%s"%(i): [U_b_D[i-1]], "decision_%s"%(i): [decision_vector[i-1]]})


	results_dict.update({"P_full_buffer": [P_full_buffer], "P_class_E": [P_class_E],
		"P_class_N": [P_class_N], "long_term_reward": [long_term_reward], 
		"throughput": [throughput], "drop_rate": [drop_rate], "tau": [tau], "class_name": [class_name],
		"threshold": [threshold], "lambda_rate": [lr]})

	df_results = pd.DataFrame(np.array(list(results_dict.values())).T, columns=list(results_dict.keys()))

	df_results.to_csv(results_path, mode='a', header=not os.path.exists(results_path))



# Solve for steady-state probabilities using the power method
def power_method(P, num_states, tol=1e-6, max_iter=1000):
	pi = np.ones(num_states) / num_states  # Initialize uniform distribution
	for _ in range(max_iter):
		next_pi = pi @ P
		if np.linalg.norm(next_pi - pi, ord=1) < tol:
			return next_pi
		pi = next_pi
	raise ValueError("Power method did not converge")


def solve_markov_chain(B, mu_E, mu_N, lambda_rate, tau, C_I, C_L):

	# Total states
	num_states = 2 * B + 1

	# Define state indices
	states = [(0, "empty")] + [(i, "E") for i in range(1, B + 1)] + [(i, "N") for i in range(1, B + 1)]
	state_index = {state: idx for idx, state in enumerate(states)}

	# Rate matrix (Q)
	Q = np.zeros((num_states, num_states))

	# Rewards array
	rewards = np.zeros(num_states)

	# Populate transition rates
	for state in states:
		idx = state_index[state]

		if state == (0, "empty"):
			# Transition from empty to (1, E) or (1, N) based on tau
			if tau == 0:
				Q[idx, state_index[(1, "E")]] = lambda_rate
			else:
				Q[idx, state_index[(1, "N")]] = lambda_rate
		elif state[1] == "E":
			i = state[0]
			if i < B:
				# Arrival transitions
				Q[idx, state_index[(i + 1, "E")]] = lambda_rate
			# Service transitions
			if i - 1 == 0:
				Q[idx, state_index[(0, "empty")]] = mu_E
			else:
				if i - 1 < tau:
					Q[idx, state_index[(i - 1, "N")]] = mu_E
				else:
					Q[idx, state_index[(i - 1, "E")]] = mu_E
			# Accrue reward for service
			rewards[idx] += C_I * mu_E
		elif state[1] == "N":
			i = state[0]
			if i < B:
				# Arrival transitions
				Q[idx, state_index[(i + 1, "N")]] = lambda_rate
			# Service transitions
			if i - 1 == 0:
				Q[idx, state_index[(0, "empty")]] = mu_N
			else:
				if i - 1 < tau:
					Q[idx, state_index[(i - 1, "N")]] = mu_N
				else:
					Q[idx, state_index[(i - 1, "E")]] = mu_N
			# Accrue reward for service
			rewards[idx] += C_L * mu_N

	# Diagonal elements of Q
	for i in range(num_states):
		Q[i, i] = -np.sum(Q[i])

	# Uniformization: Building P from Q
	max_rate = max(-Q.diagonal())
	P = Q / max_rate + np.eye(num_states)


	steady_state = power_method(P, num_states)

	# Verify if steady_state * P equals steady_state
	verification_result = np.allclose(steady_state @ P, steady_state, atol=1e-6)
	if not verification_result:
		raise ValueError("Verification failed: steady_state * P does not equal steady_state")

	# Calculate probabilities
	P_full_buffer = steady_state[state_index[(B, "E")]] + steady_state[state_index[(B, "N")]]
	P_class_E = sum(steady_state[state_index[(i, "E")]] for i in range(1, B + 1))
	P_class_N = sum(steady_state[state_index[(i, "N")]] for i in range(1, B + 1))

	# Calculate rates
	throughput = mu_E * P_class_E * C_I + mu_N * P_class_N * C_L
	misclassification_rate = mu_E * P_class_E * (1 - C_I) + mu_N * P_class_N * (1 - C_L)
	drop_rate = lambda_rate * P_full_buffer

	# Integrity checks
	total_rate = throughput + misclassification_rate + drop_rate
	if not np.isclose(total_rate, lambda_rate, atol=1e-3):
		raise ValueError(f"Integrity check failed: Total rate ({total_rate:.6f}) does not equal lambda_rate ({lambda_rate:.6f})")

	steady_state_sum = np.sum(steady_state)
	if not np.isclose(steady_state_sum, 1.0, atol=1e-6):
		raise ValueError(f"Integrity check failed: Steady-state probabilities sum ({steady_state_sum:.6f}) does not equal 1")

	# Calculate long-term average reward
	long_term_reward = np.dot(steady_state, rewards)

	# Display results
	#print("Steady-state probabilities:", steady_state)
	#print(f"Probability of full buffer: {P_full_buffer:.4f}")
	#print(f"Probability of being in class E: {P_class_E:.4f}")
	#print(f"Probability of being in class N: {P_class_N:.4f}")
	#print(f"Long-term average reward: {long_term_reward:.4f}")
	#print(f"Throughput: {throughput:.4f}")
	#print(f"Misclassification rate: {misclassification_rate:.4f}")
	#print(f"Drop rate: {drop_rate:.4f}")

	return P_full_buffer, P_class_E, P_class_N, long_term_reward, throughput, misclassification_rate, drop_rate



def extract_SMDP_params(df):
	return df.mu_a.item(), df.mu_b.item(), df.packet_loss_a.item(), df.packet_loss_b.item(), df.ee_acc.item(), df.end_acc.item()  

def main(args):

	data_path = os.path.join(config.DIR_PATH, args.model_name, "results",
		"qAdaEE_params_%s_%s_branches_%s_id_%s_%s.csv"%(args.model_name, args.n_branches, args.loss_weights_type,
			args.model_id, args.dataset_name))


	df = pd.read_csv(data_path)

	class_name_list = df.class_name.unique()

	threshold_list = [0.7]

	buffer_size_list = [5]

	lr_list = [0.9, 0.95, 0.99]

	gamma = 0.9

	for B in buffer_size_list:
		results_path = os.path.join(config.DIR_PATH, args.model_name, "results",
		"qAdaEE_results_%s_%s_branches_%s_id_%s_%s_buffer_%s_full.csv"%(args.model_name, args.n_branches, args.loss_weights_type,
			args.model_id, args.dataset_name, B))

		for lr in lr_list:

			for threshold in threshold_list:

				for class_name in class_name_list:
					print("Buffer size: %s, Lambda Rate: %s, threshold: %s, class_name: %s"%(B, lr, threshold, class_name))
					df_params = df[(df.class_name == class_name) & (df.threshold == threshold)]

					mu_a, mu_b, p_a, p_b, C_I, C_L = extract_SMDP_params(df_params)


					U, U_a_A, U_b_A, U_a_D, U_b_D, decision_vector, tau = multi_operator_value_iteration(B, mu_a, mu_b, 
						p_a, p_b, gamma, lr)

					print(mu_a, mu_b, class_name, tau)


					#P_full_buffer, P_class_E, P_class_N, long_term_reward, throughput, misclassification_rate, drop_rate = solve_markov_chain(B, mu_a, mu_b, lr, tau, C_I, C_L)

					#save_QAdaEE_results(U, U_a_A, U_b_A, U_a_D, U_b_D, decision_vector, tau, class_name, threshold, lr, 
					#	B, P_full_buffer, P_class_E, P_class_N, long_term_reward, 
					#	throughput, misclassification_rate, drop_rate, results_path)

				sys.exit()

if (__name__ == "__main__"):
	# Input Arguments to configure the early-exit model .
	parser = argparse.ArgumentParser(description="Extract the confidences obtained by DNN inference for next experiments.")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech-256", "cifar10"], help='Dataset name.')

	#We here insert the argument model_name. 
	#We evalue our novel calibration method Offloading-driven Temperature Scaling in four early-exit DNN:
	#MobileNet
	parser.add_argument('--model_name', type=str, default=config.model_name, choices=["mobilenet", "alexnet"], 
		help='DNN model name (default: %s)'%(config.model_name))

	parser.add_argument('--n_branches', type=int, default=1, help='Number of side branches.')

	parser.add_argument('--model_id', type=int, default=3, help='Model_id.')

	parser.add_argument('--loss_weights_type', type=str, default="crescent", help='loss_weights_type.')

	args = parser.parse_args()

	main(args)