from model58 import AdaGAE#13
import torch
import data_loader1 as loader
from sklearn.preprocessing import normalize
import warnings
import numpy as np

warnings.filterwarnings('ignore')

dataset = loader.coil20
[X, data, labels] = loader.load_data(dataset)
'''print(data1.shape)
[data, labels] = loader.load_guanpudata()'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data = torch.Tensor(data).to(device)
X = torch.Tensor(X).to(device)
#print(X.shape)
input_dim = data.shape[2]
layers = None
if dataset is loader.USPS:
    layers = [input_dim, 128, 64]

else:
    layers = [input_dim, 64, 4]

num_clusters = np.unique(labels[1,:]).shape[0]
listitem = [1,2,3,4,5]
import time
t0 = time.time()
for max_iter in listitem:
    accs = [];
    nmis = [];
    best_performlist = []
    for lam in np.power(2.0, np.array(range(-10, 10, 2))):
        for neighbors in [5]:
            print('-----lambda={}, neighbors={}'.format(lam, neighbors))
            gae = AdaGAE(X, data, labels, layers=layers, num_neighbors=neighbors, lam=lam, max_iter=10, max_epoch=50,
                         update=True, learning_rate=0.05, inc_neighbors=5, device=device)#0.001,0.005,0.01
            acc, nmi, best_perform = gae.run()
        accs.append(acc)
        nmis.append(nmi)
        best_performlist.append(best_perform)

    gae.max_perform(best_performlist)