from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
from torch.utils.data import DataLoader
import pickle
import torch.utils.data


from collections import defaultdict
import numpy as np
import random

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
import torch
import torchvision



from torchvision.datasets import CIFAR10, CIFAR100, Omniglot

import torch.utils.data
import torchvision.transforms as transforms
import os


def make_subset_seed(data_name, num_users, class_per_user,path,seed,fixed_data):

  if not os.path.exists(path):
    os.makedirs(path)
  if fixed_data:
     fix_file = f"subset_{data_name}_{class_per_user}_{num_users}_Seed{seed}.pickle"
     #import pickle5
     with open(fix_file, 'rb') as f:
        subsets_list = pickle.load(f)
     print(f"Complete Reading Fixed Subset : {fix_file}")
  else:
     subsets_list  = gen_random_subsets(data_name, path,  num_users, class_per_user,seed)

  train_set, val_set, test_set= get_datasets(data_name, path, normalize=True)
  return train_set, val_set, test_set, subsets_list


def generate_data_loaders_from_subsets(subsets_list, bz):
	loader_params = {"batch_size": bz, "shuffle": True, "pin_memory": True, "num_workers": 0}
	dataloaders = []
	for i, subsets in enumerate(subsets_list):
		# create dataloaders from subsets
		dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets)))
	
	return dataloaders

def generate_train_task_loaders_from_subsets(subsets_list,train_way,train_shot,train_query):
	loader_params = {"shuffle": True, "pin_memory": True, "num_workers": 0}
	subsets = subsets_list[0]
	
	dataloaders = list(map(lambda x: DataLoader(l2l.data.TaskDataset(l2l.data.MetaDataset(x),
																						  task_transforms=[
																							  NWays(l2l.data.MetaDataset(x),
																									  train_way),
																							  KShots(l2l.data.MetaDataset(x),
																										train_query + train_shot),
																							  LoadData(l2l.data.MetaDataset(x)),
																							  RemapLabels(l2l.data.MetaDataset(x))
																						  ], num_tasks=-1), **loader_params),
								  subsets))
	
	return dataloaders


def generate_train_task_loaders_from_subsets_randomway(subsets_list,num_user,train_shot,train_query, random_way_list):
	loader_params = {"shuffle": True, "pin_memory": True, "num_workers": 0}
	subsets = subsets_list[0]
	
	dataloaders = list(map(lambda x, i: DataLoader(l2l.data.TaskDataset(l2l.data.MetaDataset(x),
																							  task_transforms=[
																								  NWays(l2l.data.MetaDataset(x),
																										  random_way_list[i]),
																								  KShots(l2l.data.MetaDataset(x),
																											train_query + train_shot),
																								  LoadData(l2l.data.MetaDataset(x)),
																								  RemapLabels(l2l.data.MetaDataset(x))
																							  ], num_tasks=-1), **loader_params),
								  subsets, range(num_user)))
	return dataloaders


def cinic_download():
	import subprocess
	url = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz -O ./cinic/CINIC-10.tar.gz"
	commend = "wget"
	commend1 = f"{commend} {url}"
	commend2 = "tar -zxvf ./cinic/CINIC-10.tar.gz -C ./cinic"
	
	path = "./cinic"
	if not os.path.exists(path):
		os.makedirs(path)
	
	print("Shell Commend")
	
	subprocess.call("ls", shell=True)
	if not os.path.isfile("./cinic/CINIC-10.tar.gz"):
		print("No zip File")
		print("Downloading CINIC-10 Dataset.....")
		subprocess.call(commend1, shell=True)
	else:
		print("File Exist!")
	
	# subprocess.call("!wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz -O ./cinic/CINIC-10.tar.gz", shell=True)
	print("Checking File Done")
	print("Unziping CINIC-10 Dataset.....")
	process = subprocess.Popen(commend2, shell=True, stdout=subprocess.PIPE)
	process.wait()
	print(process.returncode)
	print("Done")

def get_datasets(data_name, dataroot, normalize=True, val_size=10000):
	"""
	get_datasets returns train/val/test data splits of CIFAR10/100 datasets
	:param data_name: name of dataset, choose from [cifar10, cifar100]
	:param dataroot: root to data dir
	:param normalize: True/False to normalize the data
	:param val_size: validation split size (in #samples)
	:return: train_set, val_set, test_set (tuple of pytorch dataset/subset)
	"""
	
	if data_name == 'cifar10':
		normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
		data_obj = CIFAR10
	elif data_name == 'cifar100':
		normalization = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
		data_obj = CIFAR100
	elif data_name == "cinic":
		if not os.path.exists("./cinic/train"):
			cinic_download()
		
		cinic_mean = [0.47889522, 0.47227842, 0.43047404]
		cinic_std = [0.24205776, 0.23828046, 0.25874835]
		normalization = transforms.Normalize(cinic_mean, cinic_std)
		cinic_trans = [transforms.ToTensor()]
		if normalize:
			cinic_trans.append(normalization)
		
		cinic_transform = transforms.Compose(cinic_trans)
	
	else:
		raise ValueError("choose data_name from [ 'cifar10', 'cifar100','cinic']")
	if data_name != "cinic":
		trans = [transforms.ToTensor()]
		
		if normalize:
			trans.append(normalization)
		
		transform = transforms.Compose(trans)
		
		dataset = data_obj(
			dataroot,
			train=True,
			download=True,
			transform=transform
		)
		
		test_set = data_obj(
			dataroot,
			train=False,
			download=True,
			transform=transform
		)
		
		train_size = len(dataset) - val_size
		train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
		return train_set, val_set, test_set
	else:
		print("Function get_dataset Call ")
		
		train_set = torchvision.datasets.ImageFolder('./cinic/train', transform=cinic_transform)
		val_set = torchvision.datasets.ImageFolder('./cinic/valid', transform=cinic_transform)
		test_set = torchvision.datasets.ImageFolder('./cinic/test', transform=cinic_transform)
		
		return train_set, val_set, test_set


def get_num_classes_samples(dataset):
	"""
	extracts info about certain dataset
	:param dataset: pytorch dataset object
	:return: dataset info number of classes, number of samples, list of labels
	"""
	# ---------------#
	# Extract labels #
	# ---------------#
	if isinstance(dataset, torch.utils.data.Subset):
		if isinstance(dataset.dataset.targets, list):
			data_labels_list = np.array(dataset.dataset.targets)[dataset.indices]
		else:
			data_labels_list = dataset.dataset.targets[dataset.indices]
	else:
		if isinstance(dataset.targets, list):
			data_labels_list = np.array(dataset.targets)
		else:
			data_labels_list = dataset.targets
	classes, num_samples = np.unique(data_labels_list, return_counts=True)
	num_classes = len(classes)
	return num_classes, num_samples, data_labels_list


def gen_classes_per_node(dataset, num_users, classes_per_user=2, high_prob=0.6, low_prob=0.4):
	"""
	creates the data distribution of each client
	:param dataset: pytorch dataset object
	:param num_users: number of clients
	:param classes_per_user: number of classes assigned to each client
	:param high_prob: highest prob sampled
	:param low_prob: lowest prob sampled
	:return: dictionary mapping between classes and proportions, each entry refers to other client
	"""
	num_classes, num_samples, _ = get_num_classes_samples(dataset)
	
	# -------------------------------------------#
	# Divide classes + num samples for each user #
	# -------------------------------------------#
	assert (classes_per_user * num_users) % num_classes == 0, "equal classes appearance is needed"
	count_per_class = (classes_per_user * num_users) // num_classes
	class_dict = {}
	for i in range(num_classes):
		# sampling alpha_i_c
		probs = np.random.uniform(low_prob, high_prob, size=count_per_class)
		# normalizing
		probs_norm = (probs / probs.sum()).tolist()
		class_dict[i] = {'count': count_per_class, 'prob': probs_norm}
	
	# -------------------------------------#
	# Assign each client with data indexes #
	# -------------------------------------#
	class_partitions = defaultdict(list)
	for i in range(num_users):
		c = []
		while len(c) != classes_per_user:
			class_counts = [class_dict[i]['count'] for i in range(num_classes)]
			max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
			selected_class = np.random.choice(max_class_counts)
			if selected_class not in c:
				c.append(selected_class)
				class_dict[c[-1]]['count'] -= 1
		class_partitions['class'].append(c)
		class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])
	return class_partitions


def gen_data_split(dataset, num_users, class_partitions):
	"""
	divide data indexes for each client based on class_partition
	:param dataset: pytorch dataset object (train/val/test)
	:param num_users: number of clients
	:param class_partitions: proportion of classes per client
	:return: dictionary mapping client to its indexes
	"""
	num_classes, num_samples, data_labels_list = get_num_classes_samples(dataset)
	
	# -------------------------- #
	# Create class index mapping #
	# -------------------------- #
	data_class_idx = {i: np.where(data_labels_list == i)[0] for i in range(num_classes)}
	
	# --------- #
	# Shuffling #
	# --------- #
	for data_idx in data_class_idx.values():
		random.shuffle(data_idx)
	
	# ------------------------------ #
	# Assigning samples to each user #
	# ------------------------------ #
	user_data_idx = [[] for i in range(num_users)]
	for usr_i in range(num_users):
		for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
			end_idx = int(num_samples[c] * p)
			user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
			data_class_idx[c] = data_class_idx[c][end_idx:]
	
	return user_data_idx


# (data_name, data_path, 100, num_users,  classes_per_user)
def gen_random_loaders(data_name, data_path, num_users, bz, classes_per_user):
	"""
	generates train/val/test loaders of each client
	:param data_name: name of dataset, choose from [cifar10, cifar100]
	:param data_path: root path for data dir
	:param num_users: number of clients
	:param bz: batch size
	:param classes_per_user: number of classes assigned to each client
	:return: train/val/test loaders of each client, list of pytorch dataloaders
	"""
	loader_params = {"batch_size": bz, "shuffle": False, "pin_memory": True, "num_workers": 0}
	dataloaders = []
	datasets = get_datasets(data_name, data_path, normalize=False)
	for i, d in enumerate(datasets):
		# ensure same partition for train/test/val
		if i == 0:
			cls_partitions = gen_classes_per_node(d, num_users, classes_per_user)
			loader_params['shuffle'] = False
		usr_subset_idx = gen_data_split(d, num_users, cls_partitions)
		# create subsets for each client
		subsets = list(map(lambda x: torch.utils.data.Subset(d, x), usr_subset_idx))
		# create dataloaders from subsets
		dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets)))
	
	return dataloaders


def gen_random_task_loaders(data_name, data_path, num_users, classes_per_user):
	"""
	generates train/val/test loaders of each client
	:param data_name: name of dataset, choose from [cifar10, cifar100]
	:param data_path: root path for data dir
	:param num_users: number of clients
	:param bz: batch size
	:param classes_per_user: number of classes assigned to each client
	:return: train/val/test loaders of each client, list of pytorch dataloaders
	"""
	loader_params = {"shuffle": False, "pin_memory": True, "num_workers": 0, "batch_size": 2}
	dataloaders = []
	datasets = get_datasets(data_name, data_path, normalize=True)
	
	for i, d in enumerate(datasets):
		# ensure same partition for train/test/val
		
		cls_partitions = gen_classes_per_node(d, num_users, classes_per_user)
		loader_params['shuffle'] = False
		usr_subset_idx = gen_data_split(d, num_users, cls_partitions)
		# create subsets for each client
		subsets = list(map(lambda x: torch.utils.data.Subset(d, x), usr_subset_idx))
		# create dataloaders from subsets
		dataloaders.append(list(map(lambda x: DataLoader(l2l.data.TaskDataset(l2l.data.MetaDataset(x),
																									 task_transforms=[
																										 NWays(l2l.data.MetaDataset(x),
																												 args.train_way),
																										 KShots(l2l.data.MetaDataset(x),
																												  args.train_query + args.shot),
																										 LoadData(l2l.data.MetaDataset(x)),
																										 RemapLabels(l2l.data.MetaDataset(x)),
																									 ], num_tasks=-1), **loader_params),
											 subsets)))
	
	return dataloaders


def gen_random_subsets(data_name, data_path, num_users, classes_per_user, seed):
	"""
	generates train/val/test loaders of each client
	:param data_name: name of dataset, choose from [cifar10, cifar100]
	:param data_path: root path for data dir
	:param num_users: number of clients
	:param bz: batch size
	:param classes_per_user: number of classes assigned to each client
	:return: train/val/test loaders of each client, list of pytorch dataloaders
	"""
	loader_params = {"shuffle": False, "pin_memory": True, "num_workers": 0, "batch_size": 2}
	dataloaders = []
	datasets = get_datasets(data_name, data_path, normalize=True)
	subsets_list = []
	
	for i, d in enumerate(datasets):
		# ensure same partition for train/test/val
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		
		cls_partitions = gen_classes_per_node(d, num_users, classes_per_user)
		loader_params['shuffle'] = False
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		usr_subset_idx = gen_data_split(d, num_users, cls_partitions)
		# create subsets for each client
		subsets = list(map(lambda x: torch.utils.data.Subset(d, x), usr_subset_idx))
		# create dataloaders from subsets
		subsets_list.append(subsets)
	
	return subsets_list