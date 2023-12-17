import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

def launch_tensor_board(log_path, port, host):
   os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
   return True

def get_l2_distance(a,b):
  return ((a-b)**2).sum()**0.5

def get_vp(a,b):
   mean = (a+b)*0.5
   return mean


def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits


def pairwise_distances(a, b):
   n = a.shape[0]
   m = b.shape[0]
   logits = ((a.unsqueeze(1).expand(n, m, -1) -
               b.unsqueeze(0).expand(n, m, -1)) ** 2).sum(dim=2)
   return logits

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def triplet_batch(anchor, positive, negative, distance_function, margin, batch=1):
   triplet = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=margin)
   n_a = len(anchor)
   n_p = len(positive)
   n_n = len(negative)
   t_loss = 0
   if batch == 0:
      for p in positive:
         for n in negative:
            t_loss = t_loss + triplet(anchor, p, n)
      t_loss = t_loss / (n_p * n_n)
   
   elif batch == 1:
      for idx, p in enumerate(positive):

         t_loss = t_loss + triplet(anchor.repeat(n_n, 1), p.repeat(n_n, 1), negative)
      t_loss = t_loss / n_p
      
   elif batch > 1:
      
      start = 0
      end = start + batch
      count = 0
      while (start < n_n ):
         if end > n_n:
            end = n_n
            n_n_batch = n_n - start
         else:
            n_n_batch = batch
         for idx, p in enumerate(positive):
            t_loss = t_loss + triplet(anchor.repeat(n_n_batch, 1), p.repeat(n_n_batch, 1), negative[start:end])
         start += batch
         end += batch
         count += 1

      t_loss = t_loss / count
   elif batch < 0:
      batch_ef = -1 * batch
      total_pair = n_p * n_n
      batch_iter = (total_pair // batch_ef)
      for i in range(batch_iter+ 1):
         if i == batch_iter:
            indicies = list(range(i * batch_ef,total_pair ))
            a_n = len(indicies)
            indicies_n = [i % n_p for i in indicies]
            indicies_p = [i // n_n for i in indicies]
         else:
            indicies = list(range(i*batch_ef,(i+1)*batch_ef ))
            a_n = len(indicies)
            indicies_n = [i % n_p for i in indicies]
            indicies_p = [i // n_n for i in indicies]

         t_loss = t_loss + triplet(anchor.repeat(a_n, 1), positive[indicies_p ], negative[indicies_n ])
      t_loss = t_loss / (batch_iter+ 1)
   
   if batch > 1000:
      for idx in range((n_p // batch) + 1):
         start = idx * batch
         end = start + batch
         n_batch = batch
         if start == n_p:
            break
         if end > n_p:
            end = n_p
            n_batch = end - start
         t_loss = t_loss + n_batch * triplet(anchor.reshape(1, -1).repeat(n_batch * n_n, 1), positive[start:end],
                                             negative.repeat(n_batch * n_n, 1))
      t_loss = t_loss / n_p
   
   return t_loss


def triplet_batch_eff(anchor, positive, negative, distance_function, margin, batch=1):
   triplet = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=margin)
   n_a = len(anchor)
   n_p = len(positive)
   n_n = len(negative)
   t_loss = 0
   if batch == 0:
      for p in positive:
         for n in negative:
            t_loss = t_loss + triplet(anchor, p, n)
      t_loss = t_loss / (n_p * n_n)
   
   if batch == 1:
      for idx, p in enumerate(positive):

         t_loss = t_loss + triplet(anchor.repeat(n_n, 1), p.repeat(n_n, 1), negative)
      t_loss = t_loss / n_p
   
   if batch > 1:
   
      for idx, p in enumerate(positive):
         t_loss = t_loss + triplet(anchor.repeat(n_n, 1), p.repeat(n_n, 1), negative)
      t_loss = t_loss / n_p
   
   if batch > 1000:
      for idx in range((n_p // batch) + 1):
         start = idx * batch
         end = start + batch
         n_batch = batch
         if start == n_p:
            break
         if end > n_p:
            end = n_p
            n_batch = end - start
         t_loss = t_loss + n_batch * triplet(anchor.reshape(1, -1).repeat(n_batch * n_n, 1), positive[start:end],
                                             negative.repeat(n_batch * n_n, 1))
      t_loss = t_loss / n_p
   
   return t_loss


def meta_update_for_sampling_error(model, batch, ways,shot,train_query, metric=None, device=None, vp=0):
   if metric is None:
      metric = pairwise_distances_logits
   if device is None:
      device = model.device()
   data, labels = batch
   bound = shot + train_query
   unique_labels = labels.unique()
   # remap label
   remap_label = [torch.tensor(x) for x in range(len(unique_labels))]
   random.shuffle(remap_label)
   original_labels = torch.tensor(labels)
   for i, label in enumerate(unique_labels):
      labels[original_labels == label][:bound] = remap_label[i]
      labels[original_labels == label][bound:] = 99999
   

   # filter for sampling from client all subset
   data = data[labels < ways]
   labels = labels[labels < ways]
   
   
   # sort
   sort = torch.sort(labels)
   data = data.squeeze(0)[sort.indices].squeeze(0)
   labels = labels.squeeze(0)[sort.indices].squeeze(0)
   
   # to Device
   data = data.to(device)
   labels = labels.to(device)
   labels_origin = labels
   
   assert data.size(0) == labels.size(0)
   unique_labels = labels.unique()
   
   data_shape = data.shape[1:]
   num_support = ways * shot
   
   num_query = data.size(0) - num_support
   
   support_data = torch.empty(
      (num_support,) + data_shape,
      device=data.device,
      dtype=data.dtype,
   )
   support_labels = torch.empty(
      num_support,
      device=labels.device,
      dtype=labels.dtype,
   )
   query_data = torch.empty(
      (num_query,) + data_shape,
      device=data.device,
      dtype=data.dtype,
   )
   query_labels = torch.empty(
      num_query,
      device=labels.device,
      dtype=labels.dtype,
   )
   
   query_start = 0
   for i, label in enumerate(unique_labels):
      # filter data
      label_data = data[labels == label]
      num_label_data = label_data.size(0)
      query_shots = num_label_data - shot
      
      assert num_label_data == shot + query_shots, \
         'Only same number of query per label supported.'
      support_start = i * shot
      support_end = support_start + shot
      # query_start = query_start + query_shots
      query_end = query_start + query_shots
      
      # set value of labels
      
      support_labels[support_start:support_end].fill_(label)
      query_labels[query_start:query_end].fill_(label)
      # set value of data
      support_data[support_start:support_end].copy_(label_data[:shot])
      query_data[query_start:query_end].copy_(label_data[shot:])
      query_start = query_end
   
   assert num_support == support_labels.shape[0]
   support = model(support_data)
   print(f"Support(CE): {support.shape}")
   support = support.reshape(ways, shot, -1).mean(dim=1)  # support로 만들어진 prototypes
   if vp == 1:
      n = len(support)
      assert ways == n
      proto_v = torch.tensor([], device=device)
      for idx, c in enumerate(support):
         for i in range(idx + 1, n):
            _vp = get_vp(c, support[i])
            proto_v = torch.cat([proto_v, _vp.reshape(1, -1)])
      if n == 2:
         proto_v = proto_v.reshape(1, -1)
      # print(support.shape)
      # print(proto_v.shape)
      support = torch.cat([support, proto_v])
      # print(support.shape)
   
   query = model(query_data)
   labels = query_labels
   logits = pairwise_distances_logits(query, support)
   loss = F.cross_entropy(logits, labels)
   acc = accuracy(logits, labels)

   print(f"Query(CE): {query.shape}")
   return loss, acc, remap_label

def meta_update(model, batch, ways, shot,train_query, metric=None, device=None,vp = 0,d_info = 0):
   if metric is None:
      metric = pairwise_distances_logits
   if device is None:
      device = model.device()
   data, labels = batch
   bound = shot + train_query
   unique_labels = labels.unique()
   # remap label
   remap_label = [torch.tensor(x) for x in range(len(unique_labels))]
   random.shuffle(remap_label)
   labels = torch.tensor(labels)
   original_labels = torch.tensor(labels)
   for i, label in enumerate(unique_labels):
      if data.shape[0] > bound*ways:
         labels[(original_labels == label).nonzero(as_tuple=True)[0][:bound]]= remap_label[i]
         labels[(original_labels == label).nonzero(as_tuple=True)[0][bound:]]= 99999
      else:
         labels[original_labels == label] = remap_label[i]

         
   # filter for sampling from client all subset
   
   if (ways < len(unique_labels)) or data.shape[0] > bound*ways:
      data = data[labels < ways]
      labels = labels[labels < ways]
   # sort
   sort = torch.sort(labels)
   data = data.squeeze(0)[sort.indices].squeeze(0)
   labels = labels.squeeze(0)[sort.indices].squeeze(0)
   
   # to Device
   data = data.to(device)
   labels = labels.to(device)
   labels_origin = labels
   
   assert data.size(0) == labels.size(0)
   unique_labels = labels.unique()

   # d inforamation collect
   if d_info == 1:
      origin = model(data)
      op = torch.tensor([], device=device)
      for l in unique_labels:
         part = origin[labels == l]
         op = torch.cat([op, part.mean(dim=0).reshape(1, -1)])
      # D Calcuration
      n = len(op)
      assert ways == n
      distance = []
      for idx, c in enumerate(op):
         for i in range(idx + 1, n):
            distance.append(get_l2_distance(c, op[i]))
   
      d_avg = torch.mean(torch.stack(distance))
   else:
      d_avg = 0
   
   data_shape = data.shape[1:]
   num_support = ways * shot
   
   num_query = data.size(0) - num_support

   support_data = torch.empty(
      (num_support,) + data_shape,
      device=data.device,
      dtype=data.dtype,
   )
   support_labels = torch.empty(
      num_support,
      device=labels.device,
      dtype=labels.dtype,
   )
   query_data = torch.empty(
      (num_query,) + data_shape,
      device=data.device,
      dtype=data.dtype,
   )
   query_labels = torch.empty(
      num_query,
      device=labels.device,
      dtype=labels.dtype,
   )
   
   query_start = 0
   for i, label in enumerate(unique_labels):
      # filter data
      label_data = data[labels == label]
      num_label_data = label_data.size(0)
      query_shots = num_label_data - shot
      
      assert num_label_data == shot + query_shots, \
         'Only same number of query per label supported.'
      support_start = i * shot
      support_end = support_start + shot
      # query_start = query_start + query_shots
      query_end = query_start + query_shots
      
      # set value of labels
      
      support_labels[support_start:support_end].fill_(label)
      query_labels[query_start:query_end].fill_(label)
      # set value of data
      support_data[support_start:support_end].copy_(label_data[:shot])
      query_data[query_start:query_end].copy_(label_data[shot:])
      query_start = query_end
      
   assert num_support == support_labels.shape[0]
   support = model(support_data)
   support = support.reshape(ways, shot, -1).mean(dim=1)
   if vp == 1:
      n = len(support)
      assert ways == n
      proto_v = torch.tensor([], device=device)
      for idx, c in enumerate(support):
         for i in range(idx + 1, n):
            _vp = get_vp(c, support[i])
            proto_v = torch.cat([proto_v ,_vp.reshape(1, -1) ])
      if n == 2:
         proto_v= proto_v.reshape(1, -1)

      support = torch.cat([support, proto_v])

      
      
   
   query = model(query_data)
   labels = query_labels
   logits = pairwise_distances_logits(query, support)
   loss = F.cross_entropy(logits, labels)
   acc = accuracy(logits, labels)


   return loss, acc, remap_label, d_avg



def triplet_update(model, batch, ways, shot,train_query, metric=None, device=None,
                  global_d=None, margin=None,margin_hp=None, t_batch=1, remap_label=None, d_from="select",version = "cvpr"):


   if metric is None:
      metric = pairwise_distances_logits
   if device is None:
      device = model.device()
   data, labels = batch
   bound = shot + train_query
   unique_labels = labels.unique()
   # remap label
   if remap_label is None:
      remap_label = [torch.tensor(x) for x in range(len(unique_labels))]
      random.shuffle(remap_label)
   else:
      remap_label = remap_label
   original_labels = torch.tensor(labels)
   labels = torch.tensor(labels)

   for i, label in enumerate(unique_labels):
      if data.shape[0] > bound * ways:
         labels[(original_labels == label).nonzero(as_tuple=True)[0][:bound]] = remap_label[i]
         labels[(original_labels == label).nonzero(as_tuple=True)[0][bound:]] = 99999
      else:
         labels[original_labels == label] = remap_label[i]
      
   # filter
   if (ways < len(unique_labels)) or data.shape[0] > bound*ways:
      data = data[labels < ways]
      labels = labels[labels < ways]
   # sort
   sort = torch.sort(labels)
   data = data.squeeze(0)[sort.indices].squeeze(0)
   labels = labels.squeeze(0)[sort.indices].squeeze(0)

   data = data.to(device)
   labels = labels.to(device)
   labels_origin = labels

   assert data.size(0) == labels.size(0)
   unique_labels = labels.unique()
   data_shape = data.shape[1:]
   num_support = ways * shot
   
   origin = model(data)
   num_query = data.size(0) - num_support
   op = torch.tensor([], device=device)
   for l in unique_labels:
      part = origin[labels == l]
      op = torch.cat([op, part.mean(dim=0).reshape(1, -1)])
   
   verbose = False
   #D Calcuration
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
   if version == "vp_activated":
      if d_from == "local":
         margin = d
      elif d_from == "global":
         margin = global_d
      elif d_from == "select":
         if d < global_d:
            margin = 0
         else:
            margin = global_d - d
      elif d_from == "margin":
         margin = margin
      else:
         raise NotImplementedError("d_from error")
   else:
      #case: cvpr
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
   margin = margin + margin_hp
   for idx, l in enumerate(labels_origin.unique()):
      anchor = op[l].reshape(1, -1)
      positive_all = origin[labels_origin == l]
      n = origin[labels_origin != l]
      t_loss = t_loss + triplet_batch(anchor, positive_all, n, distance_function=pairwise_distances,
                                      margin=margin, batch=t_batch)
   t_loss = t_loss / len(labels_origin.unique())
   
   if verbose:
      print(f"Triplet Loss :{t_loss}")
   
   return t_loss, d


def proto_eval(num_users, train_subset_loaders, test_subset_loaders, model, device):

   total_correct = []
   total_samples = []
   total_acc = []
   #Prototypes from Train set
   protos = []
   labels = []
   
   model.eval()
   
   for client in range(num_users):
      embeddings = torch.tensor([])
      train_labels = torch.tensor([])
      train_data_loader = train_subset_loaders[client]
      
      for i, samples in enumerate(train_data_loader):
         data, label = samples
         train_labels = torch.cat([train_labels, label])
         
         data = data.to(device)
         label = label.to(device)
         raw = model(data)
         batch_output = raw.reshape(raw.shape[0], -1).cpu().detach()

         try:
            embeddings = torch.cat([embeddings, batch_output])
         except:
            batch_output = raw.reshape(1, -1).cpu().detach()
            embeddings = torch.cat([embeddings, batch_output])
      label = train_labels.unique()
      proto = torch.tensor([])
      for class_idx in label:
         idx = np.where(train_labels == class_idx)[0]
         cent = embeddings[idx].mean(dim=0).reshape(1, -1)
         proto = torch.cat([proto, cent])
      proto = proto.to(device)
      protos.append(proto)
      labels.append(label)
   #Prototypes from Test set

   
   for client in range(num_users):
      test_embedding = torch.tensor([])
      test_label = torch.tensor([])
      test_data_loader = test_subset_loaders[client]
      
      for i, samples in enumerate(test_data_loader):
         data, label = samples
         test_label = torch.cat([test_label, label])
         
         data = data.to(device)
         label = label.to(device)
         raw = model(data)
         batch_output = raw.reshape(raw.shape[0], -1).cpu().detach()

         try:
            test_embedding = torch.cat([test_embedding, batch_output])
         except:
            batch_output = raw.reshape(1, -1).cpu().detach()
            test_embedding = torch.cat([test_embedding, batch_output])

      test_embedding = test_embedding.to(device)
      logits = pairwise_distances_logits(test_embedding, protos[client])
      p = logits.argmax(dim=1).view(test_label.shape)
      pred_label = labels[client][p]
      correct = (pred_label == test_label).sum()
      
      # total_sample
      sample = test_label.shape[0]
      total_samples.append(sample)
      # total_correct
      total_correct.append(correct)
      # total_acc
      acc = correct / sample
      total_acc.append(acc)
   
   mean_acc = np.mean([tc.cpu() for tc in total_acc])
   t_correct = sum([c for c in total_correct])
   t_samples = sum([t for t in total_samples])
   avg_acc = t_correct / t_samples
   model.train()
   return avg_acc, mean_acc



def get_dmin(points, n):
   distance = []
   for idx, c in enumerate(points):
      for i in range(idx + 1, n):
         distance.append(get_l2_distance(c, points[i]))
   assert len(distance) == sum([i for i in range(1, n)])
   try:
      d_min = np.min(np.array(distance))
   except:
      # print("TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu()")
      d_min = np.min(np.array([d.item() for d in distance]))
   
   return d_min


def get_dmax(points, n):
   distance = []
   for idx, c in enumerate(points):
      for i in range(idx + 1, n):
         distance.append(get_l2_distance(c, points[i]))
   assert len(distance) == sum([i for i in range(1, n)])
   d_max = np.max(np.array(distance))
   return d_max


def get_davg(points, n):
   distance = []
   for idx, c in enumerate(points):
      for i in range(idx + 1, n):
         distance.append(get_l2_distance(c, points[i]))
   assert len(distance) == sum([i for i in range(1, n)])
   try:
      d_avg = np.mean(np.array(distance))
   except:
      d_avg = np.mean(np.array([d.item() for d in distance]))
   
   return d_avg


def generate_vp(points, dim, d_min, option, device):
   vip = torch.tensor([], device=device)
   for idx, p in enumerate(points):
      if option == "uniform":
         noise_percent = torch.rand(dim) ** 2
      elif option == "standard":
         noise_percent = torch.randn(dim) ** 2
      
      normalized_percent = (noise_percent / torch.sum(noise_percent)) ** 0.5
      noise = torch.tensor([x * np.random.choice([1, -1]) * d_min for x in normalized_percent], device=device)
      
      vp = p + noise
      vp = vp.view(1, -1)
      vip = torch.cat([vip, vp])
   return vip


def d_eval(model, batch, metric=None, device=None):
   if metric is None:
      metric = pairwise_distances_logits
   if device is None:
      device = model.device()
   data, labels = batch
   unique_labels = labels.unique()
   # sort
   sort = torch.sort(labels)
   data = data.squeeze(0)[sort.indices].squeeze(0)
   labels = labels.squeeze(0)[sort.indices].squeeze(0)
   # to Device
   data = data.to(device)
   labels = labels.to(device)
   assert data.size(0) == labels.size(0)
   unique_labels = labels.unique()
   
   origin = model(data)
   op = torch.tensor([], device=device)
   for l in unique_labels:
      part = origin[labels == l]
      # print(part.shape)
      op = torch.cat([op, part.mean(dim=0).reshape(1, -1)])
   assert op.shape[0] == len(unique_labels)
   d = get_davg(op, op.shape[0])
   return d


def d_cluster_eval(model, batch, metric=None, device=None):
   if metric is None:
      metric = pairwise_distances_logits
   if device is None:
      device = model.device()
   data, labels = batch
   unique_labels = labels.unique()
   # remap label
   remap_label = [torch.tensor(x) for x in range(len(unique_labels))]
   random.shuffle(remap_label)
   original_labels = torch.tensor(labels)
   for i, label in enumerate(unique_labels):
      labels[original_labels == label] = remap_label[i]
   # sort
   sort = torch.sort(labels)
   data = data.squeeze(0)[sort.indices].squeeze(0)
   labels = labels.squeeze(0)[sort.indices].squeeze(0)
   # to Device
   data = data.to(device)
   labels = labels.to(device)
   assert data.size(0) == labels.size(0)
   unique_labels = labels.unique()
   origin = model(data)
   op = torch.tensor([], device=device)
   for l in unique_labels:
      part = origin[labels == l]
      # print(part.shape)
      op = torch.cat([op, part.mean(dim=0).reshape(1, -1)])
   assert op.shape[0] == len(unique_labels)

   logits = pairwise_distances_logits(origin, op)
   d_cluster = 0
   for idx, l in enumerate(labels):
      query_distance = abs(logits[idx, l]) ** 0.5
      d_cluster = d_cluster + query_distance
   d_cluster = d_cluster / len(labels)
   return d_cluster

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
