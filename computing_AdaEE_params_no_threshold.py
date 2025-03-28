import config, ee_dnns, sys, utils
import argparse, logging, os, torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd


def computing_service_rate(df_edge, df_cloud, class_name): # calculates service rate for each class

	# defining dataframes for classification samples:
	df_ee_classified_edge = df_edge
	df_end_classified_edge = df_edge
	df_end_classified_cloud = df_cloud

	# time metrics:
	avg_ee_edge_inf_time = df_ee_classified_edge.delta_inf_time_branch_1.mean()
	df_end_cloud_inf_time = df_end_classified_edge.delta_inf_time_branch_1 + df_end_classified_cloud.delta_inf_time_branch_2
	avg_end_cloud_inf_time = df_end_cloud_inf_time.mean()

	# calculating and returning the service rates:
	mu_a, mu_b = float(1)/float(avg_ee_edge_inf_time), float(1)/float(avg_end_cloud_inf_time) 

	service_rate_results = {"mu_a": [mu_a], "mu_b": [mu_b], "ee_edge_inf_time": [avg_ee_edge_inf_time], 
	"end_cloud_inf_time": [avg_end_cloud_inf_time], "class_name": [class_name]}

	return service_rate_results

def computing_packet_loss(df_edge, df_cloud, class_name): # calculates total packet loss

	# defining dataframes for classified samples: 
	df_ee_classified_edge = df_edge
	df_end_classified_cloud = df_cloud

	# calculating accuracies, losses and returning results
	ee_acc = float(df_ee_classified_edge.correct_branch_1.sum())/float(df_ee_classified_edge.shape[0])
	end_acc = float(df_end_classified_cloud.correct_branch_2.sum())/float(df_end_classified_cloud.shape[0])

	packet_loss_a, packet_loss_b = 1 - ee_acc, 1 - end_acc


	packet_loss_results = {"packet_loss_a": [packet_loss_a], "packet_loss_b": [packet_loss_b], 
	"ee_acc": [ee_acc], "end_acc": [end_acc], "class_name": [class_name]}

	return packet_loss_results 


def main(args):

	n_classes = 10

	inf_data_dir_path = os.path.join(config.DIR_PATH, args.model_name, "inference_data")
	os.makedirs(inf_data_dir_path, exist_ok=True)

	results_path = os.path.join(config.DIR_PATH, args.model_name, "results",
		"qAdaEE_params_%s_%s_branches_%s_id_%s_%s_no_threshold.csv"%(args.model_name, args.n_branches, args.loss_weights_type,
			args.model_id, args.dataset_name))	

	edge_inf_data_path = os.path.join(inf_data_dir_path, "inf_data_ee_%s_%s_branches_%s_id_%s_laptop_%s.csv"%(args.model_name, 
		args.n_branches, args.loss_weights_type, args.model_id, args.dataset_name))
	

	cloud_inf_data_path = os.path.join(inf_data_dir_path, "inf_data_ee_%s_%s_branches_%s_id_%s_RO_%s.csv"%(args.model_name, 
		args.n_branches, args.loss_weights_type, args.model_id, args.dataset_name))


	df_edge = pd.read_csv(edge_inf_data_path)
	df_cloud = pd.read_csv(cloud_inf_data_path)

	class_name_list = df_cloud.class_name.unique()


	for class_name in class_name_list:
		df_class_edge = df_edge[df_edge.class_name == class_name]
		df_class_cloud = df_cloud[df_cloud.class_name == class_name]

		service_rate_results = computing_service_rate(df_class_edge, df_class_cloud, class_name)
		packet_loss_results = computing_packet_loss(df_edge, df_cloud, class_name)

		service_rate_results.update(packet_loss_results)

		df_results = pd.DataFrame(np.array(list(service_rate_results.values())).T, columns=list(service_rate_results.keys()))

		df_results.to_csv(results_path, mode='a', header=not os.path.exists(results_path))



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
