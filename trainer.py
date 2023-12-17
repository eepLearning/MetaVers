import argparse
import math
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels

import datetime
import time
import copy
from tensorboardX import SummaryWriter

import timeit
from datasets import *
from util import *
from model import *
from tqdm import trange

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--max_epoch', type=int, default=1000)
	parser.add_argument('--min_epoch', type=int, default=100)
	
	# Episode
	parser.add_argument('--train_way', type=int, default=10)
	parser.add_argument('--train_shot', type=int, default=5, help="number of samples for support set at Training")
	parser.add_argument('--test_way', type=int, default=10)
	parser.add_argument('--test_shot', type=int, default=5, help="number of samples for support set at Inference")
	parser.add_argument('--train_query', type=int, default=15)
	
	parser.add_argument('--random_way', type=str, default="fixed", help="random various way learning")
	##CIFAR100-Client500
	parser.add_argument('--shot_500', type=int, default=3)
	# Experiment
	parser.add_argument('--exp_name', default="EXP_0", type=str)
	parser.add_argument('--gpu_number', default=0, type=int)
	parser.add_argument('--sd', help='default)  ', type=int, default=3)
	
	parser.add_argument('--patience', type=int, default=10)
	parser.add_argument('--train_record', default=100, type=int, help="Print Train Metrics per {train_record}")
	parser.add_argument('--evaluation_unit', type=int, default=50, help="Evaluate Performance per {evaluation_unit}")
	
	# Local Client Information
	parser.add_argument('--dataset', default="cifar100", dest='dataset', type=str,
							  choices=['cifar10', 'cifar100', 'cinic'], )
	parser.add_argument('--num_user', type=int, default=100)
	parser.add_argument('--class_per_user', type=int, default=10)
	parser.add_argument('--activated', help='default)  ', type=int, default=5)
	parser.add_argument('--control', type=str, default="fixed",help = "setting for easy commend")
	
	# Model & Train
	parser.add_argument('--method', help='default)  ', type=str, default="decentralized")
	parser.add_argument('--embedder', help='default)  ', type=str, default="lenet")
	
	parser.add_argument('--w', type=int, default=50, help="the interval value")
	parser.add_argument('--margin', type=float, default=0.0, help="margin")
	parser.add_argument('--t_batch', type=int, default=1, help="batch size for triplet:positive pair")
	parser.add_argument('--loss', type=str, default="hybrid", choices=['ce', 'triplet', 'hybrid'], help="is it hybrid loss(ce loss + triplet loss)")
	parser.add_argument('--d_from', type=str, default="select",choices=['margin', 'global', 'select'])
	parser.add_argument('--gamma', type=float, default=0.5, help="balancing hyperparameter")
	# Tuning
	parser.add_argument('--margin_hp', type=float, default=0.0, help="margin hyperparameter for tuning")
	parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
	parser.add_argument('--lr_decay', type=float, default=0.0, help="learning rate decay")
	parser.add_argument('--round_decay', type=int, default=0, help="decay every x  round")
	parser.add_argument('--optimizer', type=int, default=0, help="decay every x  round")
	parser.add_argument('--seed_list', help="seed list", nargs='+', type=int, default=[0])
	parser.add_argument('--d_info', help="ist", type=int, default=0)
	parser.add_argument('--low_way', help="lower way", type=int, default=0)
	parser.add_argument('--rand', type=int, default=1, help="control randomness, if you choose 1: fixed randomness")
	parser.add_argument('--version', type=str, default="aaai", choices=['cvpr'],
							  help="version for cvpr 2023")
	parser.add_argument('--vp', type=int, default=0, choices=[0, 1],
							  help="using virtual prototypes")
	parser.add_argument('--vp_timing', type=int, default=0,
							  help="when to start applying vp")
	

	
	
	args = parser.parse_args()
	assert args.class_per_user >= args.train_way
	assert args.class_per_user >= args.test_way
	#args.low_way = args.train_way
	
	# Data => Local Distribution
	# Configuration accorind to Dataset
	
	# CIFAR100:Client 500 Case
	if (args.dataset == "cifar100") and (args.num_user == 500):
		args.train_shot = args.shot_500
	
	if args.dataset == "cifar10":
		args.class_per_user = 2
		if args.control == "fixed":
			args.train_way = 2
			args.test_way = 2
			args.test_shot = args.train_shot
	
	if args.dataset == "cifar100":
		args.class_per_user = 10
		if args.control == "fixed":
			args.train_way = 10
			args.test_way = 10
			args.test_shot = args.train_shot
	
	if args.dataset == "cinic":
		args.class_per_user = 4
		if args.control == "fixed":
			args.train_way = 4
			args.test_way = 4
			args.test_shot = args.train_shot
	
	if args.loss != "hybrid":
		args.w = -1
	
	if args.low_way != 0:
		args.train_way = args.low_way
		args.test_way = args.low_way
	if args.activated == 0:
		args.activated  = 0.1 * args.num_user
	
	args.sd = len(args.seed_list)
	print(args)
	
	# gpu set
	GPU_NUM = args.gpu_number
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_NUM}"
	
	# seed
	final_record = []
	final_record_epoch = []
	start_time_zero = timeit.default_timer()
	if args.rand == 1:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	#original_code: for seed in range(args.sd):
	if len(args.seed_list) > 0:
		seed_list = args.seed_list
		
	else:
		seed_list = list(range(args.sd))
		
	for seed in seed_list:
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		cuda = True
		if cuda and torch.cuda.device_count():
			torch.cuda.manual_seed(seed)
			device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
		print('Current cuda device ', torch.cuda.current_device())
		
		# Generate Subset & Loader
		path = "./data/subset"
		train_set, val_set, test_set, subsets_list = make_subset_seed(args.dataset, args.num_user,
																						  args.class_per_user,
																						  path, seed, False)
		
		train_subset_loaders, val_subset_loaders, test_subset_loaders = generate_data_loaders_from_subsets(subsets_list,
																																			bz=450)
		print(f"Seed{seed} Dataset Loader Loading Done")
		
		if args.random_way != "fixed":
			random_way_list = np.random.choice(range(2, args.train_way + 1), args.num_user, True)
			train_task_loaders = generate_train_task_loaders_from_subsets_randomway(subsets_list, args.num_user,
																											args.train_shot, args.train_query,
																											random_way_list)
			print("random various way")
			print(random_way_list)
		else:
			train_task_loaders = generate_train_task_loaders_from_subsets(subsets_list, args.train_way, args.train_shot,
																							  args.train_query)
		print(f"Seed{seed} Task Loader Loading Done")
		
		if args.method == "centralized":
			print(len(train_set))
			print(args.train_way)
			print("Centralized Learning Method")
			train_dataset = l2l.data.MetaDataset(train_set)
			train_transforms = [
				NWays(train_dataset, args.train_way),
				KShots(train_dataset, args.train_query + args.train_shot),
				LoadData(train_dataset),
				RemapLabels(train_dataset),
			]
			train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms)
			train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)
			train_task_loaders = train_loader
		else:
			print("Decentralized Learning Method")
		
		exp_detail = f"Seed={str(seed)}_Dataset_{args.dataset}_Client={str(args.num_user)}_Loss={args.loss}_Margin={str(args.margin)}_From={args.d_from}_Gamma={args.gamma}_Warm={args.w}_Opti={args.optimizer}_Lr={args.lr}_decay={args.lr_decay}_{args.round_decay}(Ways={args.train_way}_Shot={args.train_shot}_Query={args.train_query}_MarginHP={args.margin_hp})"
		
		start_time = timeit.default_timer()
		log_path = os.path.join('./log/{}'.format(args.exp_name))
		time.sleep(random.randrange(1, 6, 1) / 10)
		log_path = os.path.join(log_path, exp_detail + "_GPU=" + str(args.gpu_number) + "_" + str(
			datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
		writer = SummaryWriter(logdir=log_path)
		print("num_activated_client : ", args.activated)
		print("device :", device)
		
		# Model Load Part
		if args.embedder == "conv4":
			model = Convnet()
		elif args.embedder == "lenet":
			model = LeEmbedding()
		elif args.embedder == "resnet18":
			stride = [2, 2]
			model = ResNet_Embedding(stride, BasicBlock, [2, 2, 2, 2])
			
		model = model.to(device)
		
		best_val_avg_acc, before_val_avg_acc = 0, 0
		best_test_acc = 0
		patience = 0
		
		#Optimizer
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) #Defaulr
		if args.optimizer == 1:
			optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
		elif args.optimizer == 2:
			optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-3, momentum=0.9)
		if args.lr_decay != 0:
			scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.round_decay, gamma=args.lr_decay)
		# args.max_epoch
		loss_ctr = 0
		n_loss = 0
		n_acc = 0
		
		all_d_average = args.margin
		most_d_min = all_d_average
		d_list = []
		
		n_c = 0
		n_t = 0
		n_d = 0
		
		global_d_list = []
		#information for anaysis
		d_c_c = []
		d_c_s = []
		# Training Code

		#step_iter = trange(args.num_steps)
		for epoch in trange(0, args.max_epoch + 1):
			
			
			if (patience > args.patience) and (epoch > args.min_epoch + 1):
				print("Patience Fail")
				break
			# Validation Evaluation
			if (epoch % args.evaluation_unit == 0) and (epoch > 0):
				model.eval()
				val_avg_acc, val_mean_acc = proto_eval(args.num_user, train_subset_loaders, val_subset_loaders, model,
																	device)
				test_avg_acc, test_mean_acc = proto_eval(args.num_user, train_subset_loaders, test_subset_loaders, model,
																	  device)
				
				val_avg_acc = val_avg_acc.item()
				test_avg_acc = test_avg_acc.item()
				
				if val_avg_acc > best_val_avg_acc:
					patience = 0
					best_val_avg_acc = val_avg_acc
					best_epoch = epoch
					best_test_acc = test_avg_acc
					best_model = copy.deepcopy(model)
				
				if before_val_avg_acc > val_avg_acc:
					patience += 1
				before_val_avg_acc = val_avg_acc
				
				#Logger for Tensorboard
				print('\n')
				
				terminate_time = timeit.default_timer()
				time_cost = (terminate_time - start_time)
				print('\n')
				print('Seed', seed)
				print(f"[GPU {args.gpu_number}], Best_Epoch : {best_epoch}")
				print('Iteration', epoch)
				print(f"Margin={str(args.margin)}")
				print(f"Loss={str(args.loss)}")
				print(f"D_from={str(args.d_from)}")
				print(f"{args.dataset}_Client={args.num_user}_Method={args.method}")
				print(
					f"Best_Test_ACC : {round(best_test_acc * 100, 4)}% , Best_Valid_ACC : {round(best_val_avg_acc * 100, 4)}% , Patience : {patience} ")
				print("Proto Valid ACC :", round(val_avg_acc * 100, 4), "%", 'Mean ACC : ', val_mean_acc * 100)
				print("Proto Test ACC :", round(test_avg_acc * 100, 4), "%", 'Mean ACC : ', test_mean_acc * 100)
				print("%f Seconds" % (time_cost))
				writer.add_scalar('Proto_valid_avg_acc', val_avg_acc, epoch)
				writer.add_scalar('Proto_valid_mean_acc', val_mean_acc, epoch)
				writer.add_scalar('Proto_test_avg_acc', test_avg_acc, epoch)
				writer.add_scalar('Proto_test_mean_acc', test_mean_acc, epoch)
				writer.add_scalar('Final_test_acc', best_test_acc, epoch)
				writer.add_scalar('Patience', patience, epoch)
			# Trin
			model.train()
			
			optimizer.zero_grad()
			activate_client = np.random.choice(args.num_user, args.activated, replace=False)
			
			for i in activate_client:
				
				# Client i
				
				if args.random_way != "fixed":
					train_way = random_way_list[i]
				else:
					train_way = args.train_way
				
				# CIFAR100- CLINET500
				
				if args.method == "decentralized":
					
					if (args.dataset == "cifar100") and (args.num_user == 500) and (args.method in ["decentralized"]):
						batch = next(iter(train_subset_loaders[i]))
					else:
						
						try:
							batch = next(iter(train_task_loaders[i]))
						except:
							batch = next(iter(train_subset_loaders[i]))

				
				elif args.method == "centralized":
					
					try:
						batch = next(iter(train_task_loaders))
					except:
						batch = next(iter(train_subset_loaders[i]))
						print(f"{args.dataset}: {args.num_user}")
						print(batch[1].shape)
				
				else:
					raise NotImplementedError("args.mehtod error")
				
				if args.loss == "hybrid":
					args.loss = ["ce","triplet"]
					args.d_info = 0
				elif args.loss == "ce":
					args.loss = ["ce"]
				elif args.loss == "triplet":
					args.loss = ["triplet"]

				if "ce" in args.loss:
					if epoch < args.vp_timing:
						vp = 0
					else:
						vp = args.vp
					
					c_loss, acc, remap_label,_ = meta_update(model,
																		batch,
																		train_way,
																		args.train_shot,
																		args.train_query,
																		metric=pairwise_distances_logits,
																		device=device,
																		vp = vp,
																		d_info = args.d_info)
					if len(args.loss) == 1:
						t_loss = 0
						if args.d_info == 1:
							d = _
						else:
							d = 0
				if "triplet" in args.loss:
					
					if len(args.loss) == 1:
						c_loss, acc = 0, 0
						remap_label = None
						
						
					t_loss, d = triplet_update(model,
														batch,
														train_way,
														args.train_shot,
														args.train_query,
														metric=pairwise_distances_logits,
														device=device,
														global_d=all_d_average,
														margin=args.margin,
														margin_hp = args.margin_hp,
														t_batch=args.t_batch,
														remap_label=remap_label,
														d_from=args.d_from,
														version = args.version)
					
				if (d is not None) and (d!=0):
					d_list.append(d.item())
				
				if args.loss != ["ce","triplet"]:
					if args.loss == ["ce"]:
						loss = c_loss
					elif args.loss == ["triplet"]:
						loss = t_loss
					else:
						raise NotImplementedError("args.loss error")
				else:
					#if args.loss == "hybrid"
					if epoch > args.w:
						assert args.gamma > 0
						assert args.gamma <= 1
						loss = args.gamma * c_loss + (1 - args.gamma) * t_loss
					else:
						loss = c_loss
				
				loss_ctr += 1
				n_loss += loss.item()
				n_acc += acc
				c, t, d = c_loss, t_loss, d
				
				if c == 0:
					c = torch.tensor([0.0])
				if t == 0:
					t = torch.tensor([0.0])
				if d == 0:
					d = torch.tensor([0.0])
				n_c += c.item()
				n_t += t.item()
				n_d += d.item()
				loss.backward()
			########################
			for p in model.parameters():
				try:
					p.grad.data.mul_(1.0 / args.activated)
				except:
					pass
			optimizer.step()
			if args.lr_decay != 0:
				scheduler.step()
			#########################
			if epoch >= args.w:
				while (len(d_list) > ((len(activate_client) * 50))):
					d_list.pop(0)
				all_d_average = np.mean(d_list)
				global_d_list.append(all_d_average)
				writer.add_scalar('all_d_average', all_d_average, epoch)
			if epoch % 50 == 0:
				print(f"Iteration {epoch}")
				print(f"all_d :{all_d_average}")  # len(d) : {len(d_list)}")
				print(f"{args.dataset}_Client={args.num_user}")
				
				
			if (epoch % 100 == 0) and (loss != "ce"):
				d_min_eval_list = []
				d_cluster_eval_list = []
				for i in range(len(train_subset_loaders)):
					batch = next(iter(train_subset_loaders[i]))
					eval_d = d_eval(model, batch, metric=pairwise_distances_logits,
										 device=device)
					d_cluster = d_cluster_eval(model, batch, metric=pairwise_distances_logits,
														device=device)
					d_min_eval_list.append(eval_d)
					d_cluster_eval_list.append(d_cluster.item())
				print(f"Iteration {epoch} , Eval_d_min : {np.mean(d_min_eval_list)}")
				print(f"Iteration {epoch} , Eval_d_cluster : {np.mean(d_cluster_eval_list)}")
				writer.add_scalar('d_c_c', np.mean(d_min_eval_list), epoch)
				writer.add_scalar('d_c_s', np.mean(d_cluster_eval_list), epoch)
				d_c_c.append(np.mean(d_min_eval_list))
				d_c_s.append(np.mean(d_cluster_eval_list))
			if epoch % 500 == 0 and epoch > 1 :
				exp_detail = f"Seed={str(seed)}_Dataset_{args.dataset}_Client={str(args.num_user)}_Loss={args.loss}_Gamma={args.gamma}_Warm={args.w}_Opti={args.optimizer}_Lr={args.lr}_(Ways={args.train_way}_Shot={args.train_shot}_Query={args.train_query})"
				
				model_path = f"./models/{args.exp_name}/{args.dataset}/{str(args.num_user)}/{args.embedder}/"
				if not os.path.exists(model_path):
					os.makedirs(model_path)
				# Save Model
				torch.save(best_model,
							  model_path + f'/{exp_detail}_Best_Acc_{round(best_test_acc * 1000)}_epoch_{epoch}.pth')
				
			
			
			if epoch % (args.train_record) == 0:
				loss_rec = n_loss / loss_ctr
				acc_res = n_acc / loss_ctr
				print('\n')
				print('Iteration', epoch)
				print("Meta Train Error :", round(loss_rec, 4))
				if args.loss != ["triplet"]:
					print("Meta Train ACC :", round(acc_res.item() * 100, 4), "%")
				writer.add_scalar('CE_loss_100', n_c / loss_ctr, epoch)
				writer.add_scalar('Triplet_loss_100', n_t / loss_ctr, epoch)
				writer.add_scalar('D', n_d / loss_ctr, epoch)
				
				writer.add_scalar('Task_train_acc_100', acc_res, epoch)
				writer.add_scalar('Task_train_loss_100', loss_rec, epoch)
				loss_ctr = 0
				n_loss = 0
				n_acc = 0
				
				n_c = 0
				n_t = 0
				n_d = 0
		
		terminate_time = timeit.default_timer()
		time_cost = (terminate_time - start_time)
		print('\n')
		print("%fSeconds" % (time_cost))
		
		model_path = "./models"
		if not os.path.exists(model_path):
			os.makedirs(model_path)
		# Save Model
		torch.save(best_model,
					  model_path + f'/{exp_detail}_Final_Acc_{round(best_test_acc * 1000)}_Final_epoch_{best_epoch}.pth')
		writer.flush()
		writer.close()
		final_record.append(best_test_acc)
		final_record_epoch.append(best_epoch)
	
	time_cost = (terminate_time - start_time_zero)
	print('\n')
	print(f"Margin={str(args.margin)}")
	print('Iteration', epoch)
	print(f"{args.dataset}_Client={args.num_user}_Method={args.method}_from_{args.d_from}loss={args.loss}")
	print(f"Experimetns [{seed+1}/ {args.sd }] Complete")
	print("%fSecond for Experiments" % (time_cost))
	print(f"{round(np.mean(final_record) * 100, 2)}±{round(np.std(final_record) * (1 / math.sqrt(args.sd)) * 100, 2)}")
	
	# Writer
	log_path = os.path.join('./log/{}'.format(args.exp_name))
	time.sleep(random.randrange(1, 6, 1) / 10)
	log_path = os.path.join(log_path, "Final Table Value", exp_detail + "_" + str(
		datetime.datetime.now().strftime("%Y-%m-%d")))
	writer = SummaryWriter(logdir=log_path)
	for idx, (acc, epoch) in enumerate(zip(final_record, final_record_epoch)):
		writer.add_scalar('Best_Test_ACC', acc, idx)
		writer.add_scalar('Best_Test_EPOCH', epoch, idx)
	
	# Test Accuracy with SEM
	writer.add_text("Table_Value",
						 f"{round(np.mean(final_record) * 100, 1)}±{round(np.std(final_record) * (1 / math.sqrt(args.sd)) * 100, 1)}")
	writer.add_text("Total Time Cost", f"{time_cost / 60} M")
	writer.flush()
	writer.close()
