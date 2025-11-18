import torch
import numpy as np
import utils5
from metrics import cal_clustering_metric, re_newpre
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
import scipy.io as scio
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import AMLG.evaluation
import torch.functional
class GAE(torch.nn.Module):
    def __init__(self, X, labels, layers=None, num_neighbors=5, learning_rate=10 ** -3,
                 max_iter=500, device=None):
        super(GAE, self).__init__()
        self.layers = layers
        if self.layers is None:
            self.layers = [X.shape[1], 256, 64]
        self.device = device
        if self.device is None:
            self.torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.X = X
        self.labels = labels
        self.num_neighbors = num_neighbors
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self._build_up()
    def max_perform(self):
        print()
    def _build_up(self):
        self.W1 = get_weight_initial([self.layers[0], self.layers[1]])
        self.W2 = get_weight_initial([self.layers[1], self.layers[2]])

    def forward(self, Laplacian):
        # sparse
        embedding = Laplacian.mm(self.X.matmul(self.W1))
        embedding = torch.nn.functional.relu(embedding)
        # sparse
        self.embedding = Laplacian.mm(embedding.matmul(self.W2))
        softmax = torch.nn.Softmax(dim=1)
        recons_w = self.embedding.matmul(self.embedding.t())
        recons_w = softmax(recons_w)
        return recons_w + 10 ** -10

    def build_loss(self, recons, weights):
        size = self.X.shape[0]
        loss = torch.norm(recons - weights, p='fro') ** 2 / size
        return loss

    def run(self):
        weights, _ = adagae1.utils5.cal_weights_via_CAN(self.X.t(), self.num_neighbors)
        _ = None
        Laplacian = adagae1.utils5.get_Laplacian_from_weights(weights)
        print('Raw-CAN:', end=' ')
        self.clustering(weights, method=2, raw=True)
        torch.cuda.empty_cache()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)
        for i in range(self.max_iter):
            optimizer.zero_grad()
            recons = self(Laplacian)
            loss = self.build_loss(recons, weights)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            if (i + 1) % 50 == 0 or i == 0:
                print('Iteration-{}, loss={}, '.format(i + 1, round(loss.item(), 5)), end=' ')
                self.clustering((recons.abs() + recons.t().abs()).detach() / 2, method=2)

    def clustering(self, weights, method=2, raw=False):
        n_clusters = np.unique(self.labels).shape[0]
        if method == 0 or method == 2:
            embedding = self.X if raw else self.embedding
            embedding = embedding.cpu().detach().numpy()
            km = KMeans(n_clusters=n_clusters).fit(embedding)
            prediction = km.predict(embedding)
            acc, nmi = cal_clustering_metric(self.labels, prediction)
            print('k-means --- ACC: %5.4f, NMI: %5.4f' % (acc, nmi), end='     ')
        if method == 1 or method == 2:
            degree = torch.sum(weights, dim=1).pow(-0.5)
            L = (weights * degree).t() * degree
            L = L.cpu()
            _, vectors = L.symeig(True)
            indicator = vectors[:, -n_clusters:]
            indicator = indicator / (indicator.norm(dim=1) + 10 ** -10).repeat(n_clusters, 1).t()
            indicator = indicator.cpu().numpy()
            km = KMeans(n_clusters=n_clusters).fit(indicator)
            prediction = km.predict(indicator)
            acc, nmi = cal_clustering_metric(self.labels, prediction)
            print('SC --- ACC: %5.4f, NMI: %5.4f' % (acc, nmi), end='')
        print('')


class AdaGAE(torch.nn.Module):
    def __init__(self, X, data, labels, layers=None, lam=0.1, num_neighbors=3, learning_rate=10 ** -3,
                 max_iter=50, max_epoch=10, update=True, inc_neighbors=2, links=0, device=None):
        super(AdaGAE, self).__init__()
        if layers is None:
            layers = [1024, 256, 64]
        if device is None:
            device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
        self.X = X
        self.data =data
        self.labels = labels
        self.lam = lam
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_epoch = max_epoch
        self.num_neighbors = num_neighbors + 1
        self.embedding_dim = layers[-1]
        self.mid_dim = layers[1]
        self.input_dim = layers[0]
        self.update = update
        self.inc_neighbors = inc_neighbors
        self.max_neighbors = self.cal_max_neighbors()
        self.links = links
        self.device = device
        #self.nums_clusters = nums_clusters
        self.iter_id = 0
        self.embedding = None
        self._build_up()
    def max_perform(self,perform_list):
        max_acc = []
        for j in range(len(perform_list)):
            max_acc.append(perform_list[j]['result']['accuracy'])
        acc_max_id = max_acc.index(max(max_acc))
        print("\t")
        print("最终结果{}".format(perform_list[acc_max_id]['result']))
    def _build_up(self):
        self.W5 = get_weight_initial([self.data.shape[0],self.embedding_dim, self.num_clusters])
        self.num_cluster = get_weight_initial([self.data.shape[0],self.num_clusters, self.num_clusters])
        self.W3 = get_weight_initial([self.data.shape[0],self.input_dim, self.num_clusters])
        self.W4 = get_weight_initial([self.data.shape[0],self.input_dim, self.mid_dim])
        #self.Q = get_weight_initial([self.data.shape[0],self.input_dim, self.mid_dim])
        #self.E = get_weight_initial1([self.data.shape[0],self.data.shape[1], self.input_dim])
        #self.num_cluster_xin = get_weight_initial([self.num_clusters,self.num_clusters])

    def cal_max_neighbors(self):
        if not self.update:
            return 0
        size = self.data.shape[1]
        self.num_clusters = np.unique(self.labels[1,:]).shape[0]

        return 1.0 * size / self.num_clusters


    def forward(self):
        """self.diwei = torch.transpose(self.Q,dim0=1,dim1=2).matmul(torch.transpose(self.data,dim0=1,dim1=2))
        self.X_xin = self.W4.matmul(self.diwei)"""

    # 进行更改图
    def update_graph(self):
        weights, raw_weights = adagae1.utils5.cal_weights_via_CAN_updata(torch.transpose(self.v,dim0=1,dim1=2),self.indices[:, :self.num_clusters],
                                                                  self.num_neighbors,
                                                                  self.links)
        weights = weights.detach()
        raw_weights = raw_weights.detach()
        Laplacian = adagae1.utils5.get_Laplacian_from_weights(weights)
        return weights, Laplacian, raw_weights

    def build_loss(self, weights, raw_weights):

        size = self.data[0].shape[0]
        loss = 0
        softmax = torch.nn.Softmax(dim=1)

        #得到干净的变量，同时进行降维
        """loss += torch.subtract(torch.transpose(self.data,dim0=1,dim1=2), torch.add(self.X_xin,torch.transpose(self.E,dim0=1,dim1=2))).square().sqrt().sum()
        #loss +=torch.square(self.E).sqrt().sum()

        #进行控制变量self.W4
        e_re = torch.zeros((self.data.shape[0],self.mid_dim,self.mid_dim))
        e = torch.eye(self.mid_dim).cuda()
        for i in range(self.data.shape[0]):
            e_re[i] = e
        loss += torch.subtract(e_re.cuda(), torch.transpose(self.W4,dim0=1,dim1=2).matmul(self.W4)).square().sqrt().sum()
"""
        # weights的拉普拉斯矩阵
        e_re1 = torch.zeros((self.data.shape[0], self.data.shape[1], self.data.shape[1]))
        e1 = torch.eye(self.data.shape[1]).cuda()
        for i in range(self.data.shape[0]):
            e_re1[i] = e1
        degree = torch.sum(weights, dim=2).pow(-0.5)
        degree = degree.unsqueeze(1)
        L = e_re1.cuda() - torch.transpose((weights * degree),dim0=1,dim1=2)*degree

        # 进行KNN，得到weights
        sort_weights, _ = weights.sort(dim=2, descending=True)
        top_ks = sort_weights[:, :,self.num_neighbors]
        top_ks = top_ks.unsqueeze(1)
        top_ks = torch.transpose(top_ks.repeat(1, size, 1),dim0 = 1,dim1 = 2)

        # 用KNN方法得到低维的特征向量
        xin_weights = torch.mul(torch.ge(weights, top_ks), weights)
        dis_mean = torch.sum(xin_weights, dim=2)
        dis_mean = dis_mean.unsqueeze(1)
        dis_sum = torch.transpose(dis_mean.repeat(1, size, 1),dim0 = 1,dim1 = 2)
        dis_xin = torch.div(xin_weights, dis_sum)
        xin_Laplacian = adagae1.utils5.get_Laplacian_from_weights(weights)
        self.diwei = xin_Laplacian.cuda().matmul(self.data.cuda().matmul(self.W3))
        self.diwei = torch.nn.functional.normalize(self.diwei)
        # 得到L的特征向量
        e, self.v = torch.symeig(L, eigenvectors=True)
        sorted, self.indices = torch.sort(e, descending=False)
        loss += torch.abs(sorted[:self.num_clusters]).sum()
        self.zonghe = 0
        self.zonghe += torch.abs(sorted[:self.num_clusters]).sum()
        self.fenshu = {}
        for o in range(self.data.shape[0]):
           self.fenshu[o] = torch.abs(sorted[o][:self.num_clusters]).sum() / (self.zonghe + 10 ** -10)
        # 得到的降维后的特征向量，进行注意机制KNN
        for j in range(self.data.shape[0]):
           self.v[j , :, self.indices[j,:self.num_clusters]] = xin_Laplacian[j].cuda().matmul(torch.matmul(self.v[j , :, self.indices[j, :self.num_clusters]].cuda(), self.num_cluster[j]))
           #self.v[j , :, self.indices[j,:self.num_clusters]] = torch.nn.functional.normalize(self.v[j , :, self.indices[j,:self.num_clusters]])
        ##进行softmax函数
        #print(weights.shape)
        recons = torch.zeros(self.data.shape[0],self.data.shape[1],self.data.shape[1])
        for g in range(self.data.shape[0]):
           distance = adagae1.utils5.distance(self.v[g, :, self.indices[g, :self.num_clusters]].t(),self.v[g , :, self.indices[g , :self.num_clusters]].t())
           recons[g] = softmax(-distance)
        #进行促进
           loss += (weights[g,:,:] * torch.log(weights[g,:,:] / recons[g,:,:].cuda() + 10 ** -10)).sum(dim=1).mean()#* 1/self.fenshu[g]

        # 进行减少轨迹上的数值
        for r in range(self.data.shape[0]):
            loss += self.lam * torch.trace(self.v[r, : , self.indices[r, :self.num_clusters]].t().matmul(L[r]).matmul(self.diwei[r])) / size * 2 * 1/self.fenshu[r]  # self.embedding
            #self.v[r, :, self.indices[r, :self.num_clusters]].matmul(self.v[r, :, self.indices[r, :self.num_clusters]].t()).matmul(self.v[r, :, self.indices[r, :self.num_clusters]].t()).matmul(self.diwei[r].t())
            loss += self.lam * torch.subtract(self.v[r, :, self.indices[r, :self.num_clusters]], self.diwei[r]).square().sqrt().mean() * 1/self.fenshu[r]
        #欧氏距离的大小
        self.weight_ai = torch.zeros((self.data.shape[0],self.data.shape[0]))
        #细化后的权重
        self.Ada_weights = torch.zeros((self.data.shape[0],self.data.shape[1],self.data.shape[1])).cuda()
        #全局进行e的次方
        self.z_xin = torch.zeros((self.data.shape[1],self.data.shape[1])).cuda()

        recons_weights_sorted, _ = recons.sort(dim=2, descending=True)
        recons_top_ks = recons_weights_sorted[:, :,self.num_neighbors]
        recons_top_ks = recons_top_ks.unsqueeze(1)
        recons_top_ks = torch.transpose(recons_top_ks.repeat(1, size, 1),dim0 = 1,dim1 = 2)
        recons_xin_weights = torch.mul(torch.ge(recons, recons_top_ks), recons)

        degree_recons_xin = torch.sum(recons_xin_weights, dim=2).pow(-0.5)
        degree_recons_xin = degree_recons_xin.unsqueeze(1)

        # print(degree.shape,weights.shape)
        L_recons = e_re1.cuda() - torch.transpose((recons_xin_weights.cuda() * degree_recons_xin.cuda()), dim0=1,dim1=2) * degree_recons_xin.cuda()
        #进行协同训练
        for d in range(self.data.shape[0]):
            for a in range(d+1,self.data.shape[0]):
                self.weight_ai[d, a] = recons_xin_weights[d].cuda().square().sqrt().sum()
                self.weight_ai[a, d] = recons_xin_weights[a].cuda().square().sqrt().sum()
                if self.weight_ai[d, a] > self.weight_ai[a, d]:
                    loss += self.lam * torch.trace(self.v[a, :, self.indices[a, :self.num_clusters]].t().matmul(L[d]).matmul(
                        self.v[a, :, self.indices[a, :self.num_clusters]])) / size* 1/self.fenshu[d]
                else:
                    loss += self.lam * torch.trace(self.v[d, :, self.indices[d, :self.num_clusters]].t().matmul(L[a]).matmul(
                        self.v[d, :, self.indices[d, :self.num_clusters]])) / size* 1/self.fenshu[a]

        #图的全局欧氏距离之和
        self.zong = self.weight_ai.sum()

        #得到细化的权重值
        self.Ada_weights_z = torch.zeros((self.data.shape[1],self.data.shape[1])).cuda()
        for y in range(self.data.shape[0]):
            self.z_xin +=torch.exp(recons[y].cuda())
        for z in range(self.data.shape[0]):
            self.Ada_weights[z] = torch.div(torch.exp(recons[z].cuda()),self.z_xin)

        #self.S得到最终的共识图
        self.Ada_weights_xin = torch.zeros((self.data.shape[1],self.data.shape[1])).cuda()
        for l in range(self.data.shape[0]):
            self.Ada_weights_xin += self.Ada_weights[l]
            self.Ada_weights_z +=torch.mul(self.Ada_weights[l],recons[l].cuda())
        self.S = torch.div(self.Ada_weights_z,self.Ada_weights_xin)


        return loss,weights



    def run(self):

        weights, raw_weights = adagae1.utils5.cal_weights_via_CAN(torch.transpose(self.data,dim0 =1,dim1= 2), self.num_neighbors, self.links)
        Laplacian = adagae1.utils5.get_Laplacian_from_weights(weights)
        Laplacian = Laplacian.to_sparse()
        torch.cuda.empty_cache()
        #print('Raw-CAN:', end=' ')
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)
        acclist = []
        nmilist = []
        losslist = []
        self.predictionlist = np.zeros([self.max_epoch,self.labels[0].shape[0]])
        self.predictiondata = np.zeros([self.max_epoch,self.labels[0].shape[0],self.num_clusters])
        for epoch in range(self.max_epoch):
            for i in range(self.max_iter):
                optimizer.zero_grad()
                self()
                loss,weights_xin = self.build_loss(weights, raw_weights)
                weights = weights.cpu()
                raw_weights = raw_weights.cpu()
                loss.requires_grad_(True)
                loss.backward(retain_graph=True)
                optimizer.step()

                weights = weights.to(self.device)
                raw_weights = raw_weights.to(self.device)

            if self.num_neighbors < self.max_neighbors:
                weights, raw_weights = adagae1.utils5.cal_weights_via_CAN(torch.transpose(self.data, dim0=1, dim1=2),self.num_neighbors, self.links)
                Laplacian = adagae1.utils5.get_Laplacian_from_weights(weights)
                #weights, Laplacian, raw_weights = self.update_graph()
                acc, nmi = self.clustering(SC=True)
                self.iter_id = self.iter_id + 1
                acclist.append(acc)
                nmilist.append(nmi)
                losslist.append(loss)
                self.num_neighbors += self.inc_neighbors
            else:
                if self.update:
                    self.num_neighbors = int(self.max_neighbors)
                    break
                recons = None
                weights = weights.cpu()
                raw_weights = raw_weights.cpu()
                torch.cuda.empty_cache()
                w, _, __ = self.update_graph()
                _, __ = (None, None)
                torch.cuda.empty_cache()
                acc, nmi = self.clustering(w, k_means=False, LFjulei=False)
                weights = weights.to(self.device)
                raw_weights = raw_weights.to(self.device)
                if self.update:
                    break
        colors = list(mcolors.CSS4_COLORS.keys())
        acc_max_id = acclist.index(max(acclist))
        prediction_max = self.predictionlist[acc_max_id]
        print(AMLG.evaluation.clustering(prediction_max.astype(np.int), self.labels[0]))
        predictiondata_max = self.predictiondata[acc_max_id]
        pred_index = re_newpre(self.labels[0], prediction_max)
        xin_index = np.argsort(prediction_max)
        tsne = TSNE(n_components=2, random_state=0)  # 将文件将为二维
        result = tsne.fit_transform(predictiondata_max[xin_index,:])
        listdata = [int(t)+1 for t in range(np.unique(self.labels[1,:]).shape[0])]
        #print(listdata)
        for i in listdata:
            plt.scatter(result[np.sort(prediction_max)==i, 0], result[np.sort(prediction_max)==i, 1], s=2, c=colors[int(i)+10],label = "%d"%i)
        plt.scatter(result[pred_index==0, 0], result[pred_index==0, 1], s=2, c=colors[self.num_clusters+15],label = "error")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("The t-SNE visualization of the afinity matrix on COIL-20 dataset")
        print(type(prediction_max.reshape(prediction_max.shape[0], )))
        plt.legend(bbox_to_anchor = (1,0),loc = 3,ncol=2,borderaxespad = 0.1)#[i for i in range(np.unique(self.labels).shape[0])]
        plt.show()
        return max(acclist), max(nmilist),AMLG.evaluation.clustering(prediction_max.astype(np.int), self.labels[0])

    def clustering(self, SC=True):
        n_clusters = np.unique(self.labels).shape[0]
        if SC:
            degree = torch.sum(self.S, dim=1).pow(-0.5)
            L = (self.S * degree).t() * degree
            L = L.cpu()
            _, vectors = L.symeig(True)
            indicator = vectors[:, -n_clusters:]
            indicator = indicator / (indicator.norm(dim=1) + 10 ** -10).repeat(n_clusters, 1).t()
            indicator = indicator.detach().numpy()
            km = KMeans(n_clusters=n_clusters).fit(indicator)
            prediction = km.predict(indicator)
            self.predictionlist[self.iter_id] = prediction
            self.predictiondata[self.iter_id] = indicator
            acc, nmi = cal_clustering_metric(self.labels[0], prediction)
            print('SC --- ACC: %5.4f, NMI: %5.4f' % (acc, nmi), end='')
            print(AMLG.evaluation.clustering(prediction, self.labels[0]))
        print('')
        return acc, nmi


def get_weight_initial(shape):
    #print(len(shape))
    bound = np.sqrt(6.0 / (shape[0] + shape[1]))
    ini = torch.rand(shape) * 2 * bound - bound
    return torch.nn.Parameter(ini, requires_grad=True)



if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')

    import data_loader as loader

    dataset = loader.MNIST_TEST
    data, labels = loader.load_data(dataset)
    mDevice = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    input_dim = data.shape[1]
    X = torch.Tensor(data).to(mDevice)
    if dataset is loader.USPS:
        layers = [input_dim, 128, 64]
    elif dataset is loader.SEGMENT:
        layers = [input_dim, 10, 7]
    else:
        layers = [input_dim, 256, 64]
    for neighbor in [5, 10, 20]:
        gae = GAE(X, labels, layers=layers, num_neighbors=neighbor, learning_rate=10 ** -3, max_iter=200,
                  device=mDevice)
        gae.run()
