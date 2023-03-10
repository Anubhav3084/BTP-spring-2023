

from FedAvg.utils import *
from FedAvg.models import *
from FedAvg.train import *
from torchvision import datasets, transforms
import random

transforms_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_data_train = datasets.MNIST('./', train=True, download=True, transform=transforms_mnist)

rounds = 100
C = 0.1
K = 100
E = 5
batch_size = 10
lr = 0.01
iid_dict = iid_partition(mnist_data_train, 100)

model = LeNet()

# set manual seed for reproducibility
seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    model.cuda()

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mnist_cnn_iid_trained = training(model, rounds, batch_size, lr, mnist_data_train, iid_dict, C, K, E)