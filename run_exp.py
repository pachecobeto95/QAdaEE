import config, ee_dnns, sys, utils
import argparse, logging, os, torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd




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


def extract_SMDP_params(df):
	return df.mu_a.item(), df.mu_b.item(), df.packet_loss_a.item(), df.packet_loss_b.item(), df.ee_acc.item(), df.end_acc.item()  

def main(args):

	data_path = os.path.join(config.DIR_PATH, args.model_name, "results",
		"qAdaEE_params_%s_%s_branches_%s_id_%s_%s.csv"%(args.model_name, args.n_branches, args.loss_weights_type,
			args.model_id, args.dataset_name))


	df = pd.read_csv(data_path)

	class_name_list = df.class_name.unique()

	threshold_list = [0.7, 0.9]

	buffer_size_list = [10]

	lr_list = [5]

	gamma = 0.05

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