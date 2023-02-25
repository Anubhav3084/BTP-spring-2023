

import numpy as np
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader, Dataset

## data partitioning

def iid_partition(dataset, clients):
    """
    IID: shuffle the data and split it between clients
    """
    num_samples_per_client = int(len(dataset) / clients)
    client_dict = {}
    samples_idxs = [i for i in range(len(dataset))]

    for i in range(clients):
        client_dict[i] = set(np.random.choice(samples_idxs, num_samples_per_client, replace=False))
        samples_idxs = list(set(samples_idxs) - client_dict[i])

    return client_dict

def non_iid_partition(dataset, clients, total_shards, shard_size, num_shards_per_client):
    """
    non-IID: sort the data by the digit label
             Divide the data into N (total_shard) shards of size S (shard_size)
             Each clients will receive X shards (num_shards_per_client)
    """
    shard_idxs = [i for i in range(total_shards)]
    client_dict = {i: np.array([], dtype='int64') for i in range(clients)}
    idxs = np.arange(len(dataset))
    data_labels = dataset.targets.numpy()

    # sort the labels
    label_idxs = np.vstack((idxs, data_labels))
    label_idxs = label_idxs[:, label_idxs[1,:].argsort()]
    idxs = label_idxs[0, :]

    # divide the data into total_shards of size shards_size
    # assign num_shards_per_client to each client
    for i in range(clients):
        rand_set = set(np.random.choice(shard_idxs, num_shards_per_client, replace=False))
        shard_idxs = list(set(shard_idxs) - rand_set)

        for rand in rand_set:
            client_dict[i] = np.concatenate((client_dict[i], idxs[rand*shard_size : (rand+1)*shard_size]), axis=0)
    
    return client_dict


## dataset
class CustomDataset(Dataset):
    def __init__(self, dataset, idxs):
        super().__init__()

        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, index):
        image, label = self.dataset[self.idxs[index]]
        return image, label



if __name__ == '__main__':

    dataset = np.random.randint(0, 10, size=(10, 2))
    print(dataset.shape, len(dataset))
    print(dataset)
    out = iid_partition(dataset, 2)
    print()
    print(out)