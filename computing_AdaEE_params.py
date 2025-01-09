import config, ee_dnns, sys, utils
import argparse, logging, os, torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd


def computing_service_rate(df_edge, df_cloud, threshold, class_name):

	


def main(args):

	n_classes = 10

	inf_data_dir_path = os.path.join(config.DIR_PATH, args.model_name, "inference_data")
	os.makedirs(inf_data_dir_path, exist_ok=True)

	edge_inf_data_path = os.path.join(inf_data_dir_path, "inf_data_ee_%s_%s_branches_%s_id_%s_laptop_%s.csv"%(args.model_name, 
		args.n_branches, args.loss_weights_type, args.model_id, args.dataset_name))
	

	cloud_inf_data_path = os.path.join(inf_data_dir_path, "inf_data_ee_%s_%s_branches_%s_id_%s_RO_%s.csv"%(args.model_name, 
		args.n_branches, args.loss_weights_type, args.model_id, args.dataset_name))


	#df_edge = pd.read_csv(edge_inf_data_path)
	df_cloud = pd.read_csv(cloud_inf_data_path)

	class_name_list = df_cloud.class_name.unique()

	threshold_list = [0.7, 0.9]

	for threshold in threshold_list:

		for class_name in class_name_list:
			#df_class_edge = df_edge[df_edge.class_name == class_name]
			df_class_cloud = df_cloud[df_cloud.class_name == class_name]

			service_rate_results = computing_service_rate(df_class_edge, df_class_cloud, threshold, class_name)




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

	parser.add_argument('--loss_weights_type', type=str, default="decrescent", help='loss_weights_type.')

	args = parser.parse_args()

	main(args)