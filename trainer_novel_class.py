import argparse
import math
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels

import datetime
import time
from tensorboardX import SummaryWriter

import timeit
from datasets import *
from util import *
from util_novel_class import *
from model import *

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels


#from tensorboardX import SummaryWriter
import time
import datetime
import random
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset

class trainDataset(Dataset):
    def __init__(self):
        self.x_data = train_x
        self.y_data = train_y

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx],self.y_data[idx]
	 
	 
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--max_epoch', type=int, default=40000)
	parser.add_argument('--dataset', default="mini", dest='dataset', type=str,
							  choices=['mini', 'cifar100', ] )
	parser.add_argument('--gpu_number', default=0, type=int)
	parser.add_argument('--patience', type=int, default=10)

	parser.add_argument('--w', type=int, default=50, help="the interval value")
	parser.add_argument('--margin', type=float, default=0.0, help="margin")
	parser.add_argument('--t_batch', type=int, default=1, help="batch size for triplet:positive pair")
	parser.add_argument('--loss', type=str, default="hybrid", choices=['ce', 'triplet', 'hybrid'],
							  help="is it hybrid loss(ce loss + triplet loss)")
	parser.add_argument('--d_from', type=str, default="select", choices=['margin', 'global', 'select'])
	parser.add_argument('--gamma', type=float, default=0.5, help="balance loss")
	
	parser.add_argument('--train_record', default=100, type=int, help="Print Train Metrics per {train_record}")
	parser.add_argument('--evaluation_unit', type=int, default=500, help="Evaluate Performance per {evaluation_unit}")
	
	args = parser.parse_args()
	dataset = args.dataset
	iteration = args.max_epoch
	GPU_NUM = args.gpu_number


	cuda = True
	seed = 42
	
	random.seed(seed)
	np.random.seed(seed)
	#print(seed)
	torch.manual_seed(seed)
	#device = torch.device('cpu')
	
	torch.cuda.manual_seed(seed)
	device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
	  
  
  #Experiment Set-up
	train_way = 2
	test_way = 2
	shot = 5
	test_shot = 5
	train_query = 15
	test_query = 15
	
	###dataset
	transf = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
	path_data = "./data"
	
	if dataset == "cifarfs":
		train_dataset = l2l.vision.datasets.CIFARFS(
			root=path_data, mode='train', download=True, transform=transf)
		
		valid_dataset = l2l.vision.datasets.CIFARFS(
			root=path_data, mode='validation', download=True, transform=transf)
		
		test_dataset = l2l.vision.datasets.CIFARFS(
			root=path_data, mode='test', download=True, transform=transf)
	elif dataset == "mini":
		valid_dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode='validation', download=True)
		test_dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode='test', download=True)
		train_dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode='train', download=True)
		#valid_dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode='validation', download=False)
		#test_dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode='test', download=False)
		#train_dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode='train', download=False)
	
	elif dataset == "cifar100":
		train_dataset, valid_dataset, test_dataset = Cifar100_partion()
		
		

	train_dataset = l2l.data.MetaDataset(train_dataset)
	valid_dataset = l2l.data.MetaDataset(valid_dataset)
	test_dataset = l2l.data.MetaDataset(test_dataset)
	print("Dataset Load Complete")
	
	num_users = 64
	
	if dataset == "mini":
		dict_users, dict_ways = mini_shard_noniid(train_dataset, num_users)
	else:
		dict_users, dict_ways = c100_shard_noniid(train_dataset, num_users)
	loader_params = {"shuffle": True, "pin_memory": True, "num_workers": 0}
	task_loader = []
	for i in range(num_users):
		train_way = dict_ways[i]
		train_shot = 5
		train_query = 15
		x = DatasetSplit(train_dataset, dict_users[i])
		task_loader.append(
			
			DataLoader(l2l.data.TaskDataset(l2l.data.MetaDataset(x),
													  task_transforms=[
														  NWays(l2l.data.MetaDataset(x),
																  train_way),
														  KShots(l2l.data.MetaDataset(x),
																	train_query + train_shot),
														  LoadData(l2l.data.MetaDataset(x)),
														  RemapLabels(l2l.data.MetaDataset(x))
													  ], num_tasks=-1), **loader_params)
		
		)
	
	print("Dataset Split Complete")
	print("Model Training Start")
	print(len(task_loader))
	
	#Model
	if dataset == "mini":
		model = Convnet()
	else:
		model = conv4_FRL()
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(
		optimizer, step_size=5000, gamma=0.5)
	
	num_activate_client = 10
	num_users = 64
	loss_ctr, n_loss, n_acc, error_cnt = 0, 0, 0, 0
	global_d_list = []
	all_d_average = 1
	d_list = []
	n_c, m_t, m_d, n_t, n_d = 0, 0, 0, 0, 0
	t_loss = 0
	alpha = 0.5
	best_valid_acc = 0
	best_test_acc = []
	d_list = []
	patience = 0
	for epoch in range(1, iteration + 1):
		
		if (patience > args.patience):
			print("Patience Fail")
			break
		
		loss_ctr, n_loss, n_acc, error_cnt = 0, 0, 0, 0
		
		model.train()
		optimizer.zero_grad()
		activate_client = np.random.choice(num_users, num_activate_client, replace=False)
		
		for i in activate_client:
			
			batch = next(iter(task_loader[i]))
			meta_train_error = 0.0
			meta_train_accuracy = 0.0
			c_loss, acc  = meta_update_frl(model,
											 batch,
											 dict_ways[i],
											 train_shot,
											 train_query,
											 metric=pairwise_distances_logits,
											 device=device,
											 dataset = dataset)
			
			loss_ctr += 1
			n_loss += c_loss.item()
			n_acc += acc.item()
			if epoch > args.w:
				while (len(d_list) > ((len(activate_client) * 50))):
					d_list.pop(0)
				all_d_average = np.mean(d_list)

				
				if dict_ways[i] > 1:
					t_loss, d = triplet_update_frl(model, batch, dict_ways[i], train_shot, train_query,
													  metric=pairwise_distances_logits, device=device,
													  global_d=all_d_average, margin=None, t_batch=1, remap_label=None,
													  d_from="select",dataset = dataset)
					d_list.append(d.item())
					loss = alpha * c_loss + (1 - alpha) * t_loss
				else:
					
					loss = alpha * c_loss

			else:
				#to get d information
				t_loss, d = triplet_update_frl(model, batch, dict_ways[i], train_shot, train_query,
												  metric=pairwise_distances_logits, device=device,
												  global_d=1, margin=0.5, t_batch=1, remap_label=None, d_from="margin",dataset = dataset)
				if d != None:
					d_list.append(d.item())

				
				loss = c_loss
			loss.backward()
		for p in model.parameters():
			p.grad.data.mul_(1.0 / num_activate_client)
		optimizer.step()
		lr_scheduler.step()
		model.eval()
		
		
		if epoch % args.train_record == 0:
			loss_rec = n_loss / loss_ctr
			acc_res = n_acc / loss_ctr
			print('\n')
			print('Iteration', epoch)
			print("Meta Train Error :", round(loss_rec, 4))
			print("Meta Train ACC :", round(acc_res * 100, 4), "%")
			loss_ctr, n_loss, n_acc, error_cnt = 0, 0, 0, 0
		
		if epoch % args.evaluation_unit == 0:
			valid_acc, valid_interval = evaluation(t=5, total_group=500, dataset=valid_dataset,device=device,model = model,dataset_name=dataset)
			if valid_acc > best_valid_acc:
				patience = 0
				best_valid_acc = valid_acc
				test_acc, test_interval = evaluation(t=5, total_group=1000, dataset=test_dataset,device=device,model = model,dataset_name=dataset)
				best_test_acc.append((epoch, test_acc, test_interval))
				
			else:
				patience = patience+ 1
				print(f"patience {patience}/{args.patience}")
	
	print(f"Teriminate {epoch}")
	print(args.dataset)
	print(best_test_acc)
	print(f"Test Acc: {best_test_acc[-1]}")