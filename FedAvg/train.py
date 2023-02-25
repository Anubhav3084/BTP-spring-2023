

from FedAvg.utils import *


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
def training(model, rounds, batch_size, lr, ds, client_dict, C, K, local_epochs, plt_title, plt_color):

    global_weights = model.state_dict()
    train_loss = []
    start = time.time()

    for curr_round in range(rounds):
        w, local_loss = [], []

        m = max(int(C * K), 1)
        S_t = np.random.choice(range(K), m, replace=False)   # random set of m machines

        for k in S_t:
            local_update = ClientUpdate(ds, batch_size, lr, local_epochs, client_dict[k])
            weights, loss = local_update.train(model=copy.deepcopy(model))

            w.append(copy.deepcopy(weights))
            local_loss.append(copy.deepcopy(loss))

        # update the global weights
        weights_avg = copy.deepcopy(w[0])   # select the 1st
        for k in weights_avg.keys():
            for i in range(1, len(w)):
                weights_avg[k] += w[i][k]   # add others
            weights_avg = torch.div(weights_avg[k], len(w))
        global_weights = weights_avg

        # move the updated weights to our model state dict
        model.load_state_dict(global_weights)

        # loss
        loss_avg = sum(local_loss) / len(local_loss)
        print(f"Round: {curr_round}\tAverage loss: {round(loss_avg, 3)}")

    end = time.time()