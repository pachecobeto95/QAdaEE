import config, ee_dnns, sys, utils
import argparse, logging, os, torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd

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


	return U, U_a_A, U_b_A, U_a_D, U_b_D, decision_vector


def save_QAdaEE_results(U, U_a_A, U_b_A, U_a_D, U_b_D, decision_vector, B, results_path):

	results_dict = {}

	for i in range(1, B + 1):
		results_dict.update({"U_%s"%(i): [U[i-1]], "U_a_A_%s"%(i): [U_a_A[i-1]], "U_a_D_%s"%(i): [U_a_D[i-1]],
			"U_b_D_%s"%(i): [U_b_D[i-1]], "decision_%s"%(i): [decision_vector[i-1]]})


	df_results = pd.DataFrame(np.array(list(results_dict.values())).T, columns=list(results_dict.keys()))

	df_results.to_csv(results_path, mode='a', header=not os.path.exists(results_path))


def extract_SMDP_params(df):
	return df.mu_a.item(), df.mu_a.item(), df.packet_loss_a.item(), df.packet_loss_b.item() 

def main(args):

	data_path = os.path.join(config.DIR_PATH, args.model_name, "results",
		"qAdaEE_params_%s_%s_branches_%s_id_%s_%s_no_threshold.csv"%(args.model_name, args.n_branches, args.loss_weights_type,
			args.model_id, args.dataset_name))


	df = pd.read_csv(data_path)

	class_name_list = df.class_name.unique()

	buffer_size_list = np.arange(1, 11)

	lr_list = [0.9, 0.95, 0.99]

	gamma = 0.9

	for B in buffer_size_list:
		results_path = os.path.join(config.DIR_PATH, args.model_name, "results",
		"qAdaEE_results_%s_%s_branches_%s_id_%s_%s_buffer_%s_no_threshold.csv"%(args.model_name, args.n_branches, args.loss_weights_type,
			args.model_id, args.dataset_name, B))

		for lr in lr_list:

			for class_name in class_name_list:
				df_params = df[(df.class_name == class_name)]

				mu_a, mu_b, p_a, p_b = extract_SMDP_params(df_params)

				U, U_a_A, U_b_A, U_a_D, U_b_D, decision_vector = multi_operator_value_iteration(B, mu_a, mu_b, 
					p_a, p_b, gamma, lr)

				save_QAdaEE_results(U, U_a_A, U_b_A, U_a_D, U_b_D, decision_vector, B, results_path)


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