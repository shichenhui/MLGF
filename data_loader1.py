import scipy.io as scio
import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.io as scio
from sklearn.manifold import TSNE
from metrics import cal_clustering_metric,re_newpre
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE#进行降维
import os
import sys
import numpy as np
import torch as th
import torchvision
import torchvision.transforms as transforms

MNIST_TEST = 'mnist_test'
UMIST = 'UMIST'
COIL20 = 'COIL20'
JAFFE = 'JAFFE'
PALM = 'Palm'
USPS = 'USPSdata_20_uni'
SEGMENT = 'segment_uni'
NEWS = '20news_uni'
TEXT = 'text1_uni'
ISOLET = 'Isolet'
coil20 = r'D:\shichenhui\others\1小组文件\辛永杰\S202120110711-辛永杰_研究生成果_毕业存留\dataset\data_COIL20.mat'
coil100 = 'COIL100'
orl = "ORL"
#进行数据保存
'''def export_dataset(name, views, labels):
    processed_dir = "data/"
    os.makedirs(processed_dir, exist_ok=True)
    file_path = processed_dir+ f"{name}.mat"
    npz_dict = {"labels": labels, "n_views": len(views)}
    for i, v in enumerate(views):
        npz_dict[f"view_{i}"] = v
    np.savez(file_path, **npz_dict)
#进行数据处理
def coil():
    from skimage.io import imread

    data_dir = "data/COIL20xin1.mat"
    img_size = (1, 128, 128)
    n_objs = 20
    n_imgs = 72
    n_views = 3
    assert n_imgs % n_views == 0

    n = (n_objs * n_imgs) // n_views

    imgs = np.empty((n_views, n, *img_size))
    labels = []

    img_idx = np.arange(n_imgs)

    for obj in range(n_objs):
        obj_img_idx = np.random.permutation(img_idx).reshape(n_views, n_imgs // n_views)
        labels += (n_imgs // n_views) * [obj]

        for view, indices in enumerate(obj_img_idx):
            for i, idx in enumerate(indices):
                fname = data_dir + f"obj{obj + 1}__{idx}.png"
                img = imread(fname)[None, ...]
                imgs[view, ((obj * (n_imgs // n_views)) + i)] = img

    assert not np.isnan(imgs).any()
    views = [imgs[v] for v in range(n_views)]
    labels = np.array(labels)
    export_dataset("coil", views=views, labels=labels)'''



def load_cora():
    path = 'data/cora.mat'
    data = scio.loadmat(path)
    labels = data['gnd']
    labels = np.reshape(labels, (labels.shape[0],))
    X = data['fea']
    print(X)
    X = X.astype(np.float32)
    print(X.shape[1])
    X /= np.max(X)
    links = data['W']
    return X, labels, links


def load_data(name):
    path = name
    data = scio.loadmat(path)
    print(data)
    #print(data.shape)
    labels = data['Y'].T
    print(type(labels))
    labels = np.reshape(labels, (labels.shape[0],))
    X = data['X']
    print(X.shape)
    #print(X.shape)
    n_view = 2
    """data_choose = np.zeros((int(X.shape[0]/4),X.shape[1]))
    label_choose = np.zeros((int(labels.shape[0] / 4)))
    for e in range(X.shape[0]):
        if e%4==0:
            data_choose[e//4] = X[e]
            label_choose[e//4] = labels[e]
    print(data_choose.shape,label_choose.shape)"""
    data_np = np.zeros((n_view,int(X.shape[0]/n_view),X.shape[1]))
    label_np = np.zeros((n_view,int(X.shape[0]/n_view)))
    for j in range(X.shape[0]):
        u = j % n_view
        w = j // n_view
        data_np[u, w] =X[j]
        label_np[u, w] = labels[j]

    #print(data_np.shape,label_np.shape)
    #print(X.shape[0],X.shape[1])
    data_np = data_np.astype(np.float32)
    data_np /= np.max(data_np)
    #print(label_np[1,:])
    X = X.astype(np.float32)
    X /= np.max(X)
    print(data_np,label_np)
    return X,data_np, label_np

def load_guanpudata():
    path = 'data/star_AFGK_2kx4.csv'
    my_data = genfromtxt(path, delimiter=',')
    data = normalize(my_data)
    pca = PCA(n_components=100)  # 降到100维
    data = pca.fit_transform(data)  # 用pca模型来训练,并降到100维
    data_sub = []
    lable = [0,1,2,3]
    n = int(my_data.shape[0]/len(lable))
    lablelist = [i for i in range(int(data.shape[0]/10))]
    for i in range(len(lable)):
        for j in range(int(n/10)):
            data_sub.append(data[i*n+j*10])
            lablelist[int(n/10)*i+j]= lable[i]
    lablelist = np.array(lablelist)
    data_sub = np.array(data_sub)
    km = KMeans(n_clusters=len(lable)).fit(data_sub)
    prediction = km.predict(data_sub)
    pred_index = re_newpre(lablelist, prediction)
    acc, nmi = cal_clustering_metric(lablelist, prediction)
    tsne = TSNE(n_components=2, random_state=0)  # 将文件将为二维
    result = tsne.fit_transform(data_sub)
    color = ['cornflowerblue', 'gold', "gray", "violet"]
    for i in range(len(prediction)):
        if pred_index[i] == 1:
            plt.scatter(result[i, 0], result[i, 1], s=0.7, c=color[lablelist[i]])
        else:
            plt.scatter(result[i, 0], result[i, 1], s=0.7, c='red')
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.legend((A, F, G, K, incorrect), ("A", "F", "G", "K","False"))  # 图片标签
    plt.title('k-means --- ACC: %5.4f, NMI: %5.4f' % (acc, nmi))
    plt.show()
    return data_sub,lablelist

if __name__ == '__main__':
    load_data("100leaves")
    #print(load_guanpudata())
    #data,lablelist = load_guanpudata()
    #print(data.shape,lablelist.shape)
    #coil()