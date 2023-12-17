import random
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset
import torch
import numpy as np
import argparse
import math
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels

import datetime
import time
from tensorboardX import SummaryWriter

import timeit
from datasets import *
from util import *
from sklearn.model_selection import KFold

import torch

from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)
	 
	 
def Cifar100_partion():
	# CIFAR100 : Train/Valid/Test Partion for disjoint class set-up
	trans = [transforms.ToTensor()]
	transform = transforms.Compose(trans)
	dataroot = "./data"
	normalization = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
	data_obj = CIFAR100
	trainset = data_obj(
		dataroot,
		train=True,
		download=True,
		transform=transform
	)
	
	testset = data_obj(
		dataroot,
		train=False,
		download=True,
		transform=transform
	)
	
	all_data = np.concatenate((trainset.data, testset.data), axis=0)
	all_label = np.concatenate((trainset.targets, testset.targets), axis=0)
	train_mask = []
	test_mask = []
	valid_mask = []
	for label in all_label:
		if label in [x for x in range(0, 64)]:
			train_mask.append(True)
		else:
			train_mask.append(False)
		if label in [x for x in range(64, 84)]:
			test_mask.append(True)
		else:
			test_mask.append(False)
		if label in [x for x in range(84, 100)]:
			valid_mask.append(True)
		else:
			valid_mask.append(False)
	
	train_x = all_data[train_mask]
	test_x = all_data[test_mask]
	
	train_y = all_label[train_mask]
	test_y = all_label[test_mask]
	
	valid_y = all_label[valid_mask]
	valid_x = all_data[valid_mask]
	
	
	class trainDataset(Dataset):
		def __init__(self):
			self.x_data = train_x
			self.y_data = train_y
		
		def __len__(self):
			return len(self.x_data)
		
		def __getitem__(self, idx):
			return self.x_data[idx], self.y_data[idx]
	
	
	class testDataset(Dataset):
		def __init__(self, test_x, test_y):
			self.x_data = test_x
			self.y_data = test_y
		
		def __len__(self):
			return len(self.x_data)
		
		def __getitem__(self, idx):
			return self.x_data[idx], self.y_data[idx]
	
	train_dataset = trainDataset()
	test_dataset = testDataset(test_x, test_y)
	valid_dataset = testDataset(valid_x, valid_y)
	
	return train_dataset,valid_dataset,test_dataset


def mini_shard_noniid(dataset, num_users):
	"""
	Sample non-I.I.D client data from CIFAR10 dataset
	:param dataset:
	:param num_users:
	:return:
	"""
	num_shards, num_imgs = 128, 300
	idx_shard = [i for i in range(num_shards)]
	dict_users = {i: np.array([]) for i in range(num_users)}
	dict_ways = {i: np.array([]) for i in range(num_users)}
	idxs = np.arange(num_shards * num_imgs)
	# labels = dataset.train_labels.numpy()
	labels = np.array(dataset.dataset.y)
	
	# sort labels
	idxs_labels = np.vstack((idxs, labels))
	idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
	idxs = idxs_labels[0, :]
	
	# divide and assign
	for i in range(num_users):
		rand_set = set(np.random.choice(idx_shard, 2, replace=False))
		idx_shard = list(set(idx_shard) - rand_set)
		for rand in rand_set:
			dict_users[i] = np.concatenate(
				(dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
	
	for i in range(num_users):
		dict_ways[i] = len(np.unique(labels[[int(d) for d in dict_users[i]]]))
	return dict_users, dict_ways


def c100_shard_noniid(dataset, num_users):
	"""
	Sample non-I.I.D client data from CIFAR10 dataset
	:param dataset:
	:param num_users:
	:return:
	"""
	num_shards, num_imgs = 128, 300
	idx_shard = [i for i in range(num_shards)]
	dict_users = {i: np.array([]) for i in range(num_users)}
	dict_ways = {i: np.array([]) for i in range(num_users)}
	idxs = np.arange(num_shards * num_imgs)
	# labels = dataset.train_labels.numpy()
	labels = np.array(dataset.dataset.y_data)
	
	# sort labels
	idxs_labels = np.vstack((idxs, labels))
	idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
	idxs = idxs_labels[0, :]
	
	# divide and assign
	for i in range(num_users):
		rand_set = set(np.random.choice(idx_shard, 2, replace=False))
		idx_shard = list(set(idx_shard) - rand_set)
		for rand in rand_set:
			dict_users[i] = np.concatenate(
				(dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
	
	for i in range(num_users):
		dict_ways[i] = len(np.unique(labels[[int(d) for d in dict_users[i]]]))
	return dict_users, dict_ways


def evaluation(dataset,t=5, total_group=1000,device=None,model=None,dataset_name=None):
	kf = KFold(n_splits=10, shuffle=True)
	# t = 5
	total_acc = []
	# total_group = 1000
	for group in range(total_group):
		if group % 50 == 0:
			print(f"Group {group}the Complete")
		

		test_users = [[] for i in range(10)]
		t_class = np.random.choice(dataset.labels, t, replace=False)
		for select_class in t_class:
			a = dataset.labels_to_indices[select_class]
			for idx, partion in enumerate(kf.split(a)):
				index = partion[1]
				test_users[idx].extend([a[i] for i in index])
		
		# evaluation
		test_acc = []
		for t_usr in test_users:
			test_client_loader = DataLoader(DatasetSplit(dataset, t_usr), batch_size=len(t_usr))
			data, labels = next(iter(test_client_loader))
			
			# remap
			unique_labels = labels.unique()
			remap_label = [torch.tensor(x) for x in range(len(unique_labels))]
			original_labels = torch.tensor(labels)
			for i, label in enumerate(unique_labels):
				labels[original_labels == label] = remap_label[i]
			
			sort = torch.sort(labels)
			data = data.squeeze(0)[sort.indices].squeeze(0)
			labels = labels.squeeze(0)[sort.indices].squeeze(0)
			
			if dataset_name == "cifar100":
				data = data.transpose(1, 3)
				data = data.to(device).float()
			else:
				data = data.to(device)
			labels = labels.to(device)
			
			unique_labels = labels.unique()
			embeddings = model(data)
			
			half = KFold(n_splits=2, shuffle=True)
			for i in half.split(np.arange(len(data))):
				support_idx = i[0]
				query_idx = i[1]
			
			support = embeddings[support_idx]
			query = embeddings[query_idx]
			query_labels = labels[query_idx]
			support_labels = labels[support_idx]
			op = torch.tensor([], device=device)
			for l in unique_labels:
				part = support[support_labels == l]
				op = torch.cat([op, part.mean(dim=0).reshape(1, -1)])
			
			logits = pairwise_distances_logits(query, op)
			acc = accuracy(logits, query_labels)
			test_acc.append(acc.item())
		avg_acc = np.mean(test_acc)
		total_acc.append(avg_acc)
	final_acc = np.mean(total_acc)
	print(f"{len(total_acc)} group acc : {final_acc}")
	interval = np.std(total_acc) * 1.96 * 1 / math.sqrt(len(total_acc))
	print("95% confidence interval :", interval)
	return final_acc, interval
def meta_update_frl(model, batch, ways, shot, query_num, metric=None, device=None,dataset=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch
    #print(data.shape)
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)
    if dataset == "cifar100":
       data = data.transpose(1,3)
       data = data.to(device).float()
    else:
       data = data.to(device)
    labels = labels.to(device)
    n_items = shot * ways

    # Sort data samples by labels
    # TODO: Can this be replaced by ConsecutiveLabels ?


    # Compute support and query embeddings
    embeddings = model(data)
    #print(embeddings.shape)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc

def triplet_update_frl(model, batch, ways, shot, query_num, metric=None, device=None,
						global_d=None, margin=None, t_batch=1, remap_label=None, d_from="select",dataset = None):
	all_d_min = 0
	if metric is None:
		metric = pairwise_distances_logits
	if device is None:
		device = model.device()
	data, labels = batch
	unique_labels = labels.unique()
	# remap label
	if remap_label is None:
		remap_label = [torch.tensor(x) for x in range(len(unique_labels))]
		random.shuffle(remap_label)
	else:
		remap_label = remap_label
	original_labels = torch.tensor(labels)
	for i, label in enumerate(unique_labels):
		labels[original_labels == label] = remap_label[i]
	# filter
	if ways < len(unique_labels):
		data = data[labels < ways]
		labels = labels[labels < ways]
	# sort
	sort = torch.sort(labels)
	data = data.squeeze(0)[sort.indices].squeeze(0)
	labels = labels.squeeze(0)[sort.indices].squeeze(0)
	if dataset == "cifar100":
		data = data.transpose(1, 3)
		data = data.to(device).float()
	else:
		data = data.to(device)
	labels = labels.to(device)
	labels_origin = labels

	assert data.size(0) == labels.size(0)
	unique_labels = labels.unique()
	data_shape = data.shape[1:]
	num_support = ways * shot
	
	
	origin = model(data)
	num_query = data.size(0) - num_support
	assert num_query % ways == 0, 'Only query_shot == support_shot supported.'
	query_shots = num_query // ways
	op = origin.reshape(ways, (shot + query_shots), -1).mean(dim=1)
	
	verbose = False
	
	n = len(op)
	assert ways == n
	distance = []
	for idx, c in enumerate(op):
		for i in range(idx + 1, n):
			distance.append(get_l2_distance(c, op[i]))
	try:
		d_avg = torch.mean(torch.stack(distance))
	except:
		d_avg = None
	d = d_avg
	if d_from == "local":
		margin = d
	elif d_from == "global":
		margin = global_d
	elif d_from == "select":
		if d < global_d:
			margin = global_d
		else:
			margin = d
	elif d_from == "margin":
		margin = margin
	else:
		raise NotImplementedError("d_from error")
	
	t_loss = 0
	for idx, l in enumerate(labels_origin.unique()):
		anchor = op[l].reshape(1, -1)
		positive_all = origin[labels_origin == l]
		n = origin[labels_origin != l]
		t_loss = t_loss + triplet_batch(anchor, positive_all, n, distance_function=pairwise_distances,
												  margin=margin, batch=t_batch)
	t_loss = t_loss / len(labels_origin.unique())
	# print(margin)
	
	if verbose:
		print(f"Triplet Loss :{t_loss}")
	
	return t_loss, d