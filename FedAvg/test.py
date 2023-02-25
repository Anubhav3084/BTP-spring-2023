

from FedAvg.utils import *

def testing(model, dataset, batch_size, criterion, num_classes, classes):

    test_loss = 0.0
    correct_class = list(0. for i in range(num_classes))
    total_class = list(0. for i in range(num_classes))

    test_dl = DataLoader(dataset, batch_size)
    l = len(test_dl)
    
    model.eval()

    for x, y in test_dl:

        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        
        preds = model(x)
        loss = criterion(preds, y)
        test_loss += loss.item() * x.size(0)

        _, pred = torch.max(preds, 1)

        correct_tensor = pred.eq(y.data.view_as(pred))