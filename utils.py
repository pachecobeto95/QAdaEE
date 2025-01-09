import config
from torchvision import datasets, transforms
import torch, os, sys, requests, cv2
import numpy as np

def save_indices(train_idx, val_idx, test_idx, indices_path):

	data_dict = {"train": train_idx, "val": val_idx, "test": test_idx}
	torch.save(data_dict, indices_path)

def get_indices(dataset, split_ratio, indices_path):
	
	if (not os.path.exists(indices_path)):

		nr_samples = len(dataset)

		indices = list(torch.randperm(nr_samples).numpy())	

		train_val_size = nr_samples - int(np.floor(split_ratio * nr_samples))

		train_val_idx, test_idx = indices[:train_val_size], indices[train_val_size:]

		train_size = len(train_val_idx) - int(np.floor(split_ratio * len(train_val_idx) ))

		train_idx, val_idx = train_val_idx[:train_size], train_val_idx[train_size:]

		save_indices(train_idx, val_idx, test_idx, indices_path)

	else:
		data_dict = torch.load(indices_path)
		train_idx, val_idx, test_idx = data_dict["train"], data_dict["val"], data_dict["test"]	

	return train_idx, val_idx, test_idx


def load_cifar10(batch_size, dataset_path):
    """
    Função para carregar o dataset CIFAR-10 com transformações básicas.
    
    Args:
        batch_size (int): Tamanho do lote para os DataLoaders.
        data_dir (str): Diretório onde os dados serão armazenados ou lidos.
    
    Returns:
        tuple: (train_loader, test_loader) DataLoaders para treino e teste.
    """
    # Transformações para normalizar os dados e realizar augmentations simples
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Flip horizontal aleatório
        transforms.RandomCrop(32, padding=4),  # Crop com padding
        transforms.ToTensor(),  # Converte imagem para tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalização
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    # Carregar os datasets
    train_dataset = datasets.CIFAR10(
        root=dataset_path, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=dataset_path, train=False, download=True, transform=transform_test
    )
    
    # Criar DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    return train_loader, test_loader, test_dataset.classes



def load_caltech256(args, dataset_path, indices_path):

	mean, std = [0.457342265910642, 0.4387686270106377, 0.4073427106250871], [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

	torch.manual_seed(args.seed)

	transformations_train = transforms.Compose([
		transforms.Resize((args.input_dim, args.input_dim)),
		transforms.RandomChoice([
			transforms.ColorJitter(brightness=(0.80, 1.20)),
			transforms.RandomGrayscale(p = 0.25)]),
		transforms.CenterCrop((args.dim, args.dim)),
		transforms.RandomHorizontalFlip(p=0.25),
		transforms.RandomRotation(25),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])

	transformations_test = transforms.Compose([
		transforms.Resize((args.input_dim, args.input_dim)),
		transforms.CenterCrop((args.dim, args.dim)),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])

	# This block receives the dataset path and applies the transformation data. 
	train_set = datasets.ImageFolder(dataset_path, transform=transformations_train)
	val_set = datasets.ImageFolder(dataset_path, transform=transformations_test)
	test_set = datasets.ImageFolder(dataset_path, transform=transformations_test)

	class_names = val_set.classes
	class_names = [name[4:] for name in class_names]

	train_idx, val_idx, test_idx = get_indices(train_set, args.split_ratio, indices_path)

	train_data = torch.utils.data.Subset(train_set, indices=train_idx)
	val_data = torch.utils.data.Subset(val_set, indices=val_idx)
	test_data = torch.utils.data.Subset(test_set, indices=test_idx)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_train, 
		shuffle=True, num_workers=4, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers=4, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=4, pin_memory=True)

	return train_loader, val_loader, test_loader, class_names