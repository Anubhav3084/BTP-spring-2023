

import numpy as np
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


## training algorithms

class ClientUpdate(object):
    def __init__(self, dataset, batchsize, lr, epochs, idxs):
        
        self.train_dl = DataLoader(CustomDataset(dataset, idxs), batch_size=batchsize, shuffle=True)
        self.lr = lr
        self.epochs = epochs

    def train(self, model):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.5)
        # TODO: figure out why momentum is used
        
        epoch_loss = []

        for epoch in range(self.epochs):            
            train_loss = 0.0
            model.train()

            for x, y in self.train_dl:

                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()

                optimizer.zero_grad()
                preds = model(x)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                # update training loss
                train_loss += loss.item() * x.size(0)

            # average losses
            train_loss = train_loss / len(self.train_dl.dataset)
            epoch_loss.append(train_loss)

        total_loss = sum(epoch_loss) / len(epoch_loss)

        return model.state_dict(), total_loss
    

## server side training
def training(model, rounds, batch_size, lr, ds, data_dict, C, K, E, plt_title, plt_color):

    global_weights = model.state_dict()
    train_loss = []
    start = time.time()

    for curr_round in range(rounds):
        w, local_loss = [], []

        m = max(int(C * K), 1)

    # TODO: complete the code


if __name__ == '__main__':

    dataset = np.random.randint(0, 10, size=(10, 2))
    print(dataset.shape, len(dataset))
    print(dataset)
    out = iid_partition(dataset, 2)
    print()
    print(out)