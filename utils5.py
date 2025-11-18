import torch


def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result

'''def cal_weights_via_CAN_Sam(X ,  dinces ,num_cluster):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return:
    """
    weights_list1 = torch.zeros((X.shape[0], X.shape[2], X.shape[2]))
    weights_raw_list1 = torch.zeros((X.shape[0], X.shape[2], X.shape[2]))
    size = X[0].shape[1]
    for i in range(X.shape[0]):
        distances = distance(X[i,dinces[i,0:num_cluster],:], X[i,dinces[i,0:num_cluster],:])
        softmax = torch.nn.Softmax(dim=1)
        weights = softmax(-distances)
        raw_weights = weights
        weights = (weights + weights.t()) / 2
        raw_weights = raw_weights.cuda()
        weights = weights.cuda()
        weights_list1[i] = weights
        weights_raw_list1[i] = raw_weights

    return weights_list1,weights_raw_list1

def cal_weights_via_CAN_Dis(X , X1):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return:
    """
    size = X.shape[1]
    distances = distance(X, X1)
    softmax = torch.nn.Softmax(dim=1)
    weights = softmax(-distances)
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.cuda()
    weights = weights.cuda()


    return weights,raw_weights'''

def cal_weights_via_CAN_Dis(X , X1 , num_neighbors , links=0):
    for i in range(X.shape[0]):
        size = X.shape[1]
        # print(X[i].shape)
        distances = distance(X, X1)
        distances = torch.max(distances, torch.t(distances))
        sorted_distances, _ = distances.sort(dim=1)
        top_k = sorted_distances[:, num_neighbors]
        top_k = torch.t(top_k.repeat(size, 1)) + 10 ** -10

        sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
        sum_top_k = torch.t(sum_top_k.repeat(size, 1))
        sorted_distances = None
        torch.cuda.empty_cache()
        T = top_k - distances
        distances = None
        torch.cuda.empty_cache()
        weights = torch.div(T, num_neighbors * top_k - sum_top_k)
        T = None
        top_k = None
        sum_top_k = None
        torch.cuda.empty_cache()
        weights = weights.relu().cpu()
        if links is not 0:
            links = torch.Tensor(links).cuda()
            weights += torch.eye(size).cuda()
            weights += links
            weights /= weights.sum(dim=1).reshape([size, 1])
        torch.cuda.empty_cache()
        raw_weights = weights
        weights = (weights + weights.t()) / 2
        raw_weights = raw_weights.cuda()
        weights = weights.cuda()
    return weights, raw_weights


def cal_weights_via_CAN_updata(X,Z, num_neighbors, links=0):

    weights_list = torch.zeros((X.shape[0],X.shape[2],X.shape[2]))
    weights_raw_list = torch.zeros((X.shape[0], X.shape[2], X.shape[2]))
    for i in range(X.shape[0]):
       size = X[i].shape[1]
       #print(X[i].shape)
       distances = distance(X[i,Z[i,:],:], X[i,Z[i,:],:])
       distances = torch.max(distances, torch.t(distances))
       sorted_distances, _ = distances.sort(dim=1)
       top_k = sorted_distances[:, num_neighbors]
       top_k = torch.t(top_k.repeat(size, 1)) + 10**-10

       sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
       sum_top_k = torch.t(sum_top_k.repeat(size, 1))
       sorted_distances = None
       torch.cuda.empty_cache()
       T = top_k - distances
       distances = None
       torch.cuda.empty_cache()
       weights = torch.div(T, num_neighbors * top_k - sum_top_k)
       T = None
       top_k = None
       sum_top_k = None
       torch.cuda.empty_cache()
       weights = weights.relu().cpu()
       if links is not 0:
           links = torch.Tensor(links).cuda()
           weights += torch.eye(size).cuda()
           weights += links
           weights /= weights.sum(dim=1).reshape([size, 1])
       torch.cuda.empty_cache()
       raw_weights = weights
       weights = (weights + weights.t()) / 2
       weights_list[i] = weights
       weights_raw_list[i] = raw_weights
    weights_raw_list = weights_raw_list.cuda()
    weights_list = weights_list.cuda()

    return weights_list, weights_raw_list

def cal_weights_via_CAN_begin(X, num_neighbors, links=0):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return:
    """
    weights_list = torch.zeros((X.shape[0],X.shape[1],X.shape[1]))
    weights_raw_list = torch.zeros((X.shape[0], X.shape[1], X.shape[1]))
    for i in range(X.shape[0]):
       size = X[i].shape[1]
       #print(X[i].shape)
       distances = distance(X[i], X[i])
       distances = torch.max(distances, torch.t(distances))
       sorted_distances, _ = distances.sort(dim=1)
       top_k = sorted_distances[:, num_neighbors]
       top_k = torch.t(top_k.repeat(size, 1)) + 10**-10

       sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
       sum_top_k = torch.t(sum_top_k.repeat(size, 1))
       sorted_distances = None
       torch.cuda.empty_cache()
       T = top_k - distances
       distances = None
       torch.cuda.empty_cache()
       weights = torch.div(T, num_neighbors * top_k - sum_top_k)
       T = None
       top_k = None
       sum_top_k = None
       torch.cuda.empty_cache()
       weights = weights.relu().cpu()
       if links is not 0:
           links = torch.Tensor(links).cuda()
           weights += torch.eye(size).cuda()
           weights += links
           weights /= weights.sum(dim=1).reshape([size, 1])
       torch.cuda.empty_cache()
       raw_weights = weights
       weights = (weights + weights.t()) / 2
       weights_list[i] = weights
       weights_raw_list[i] = raw_weights
    weights_raw_list = weights_raw_list.cuda()
    weights_list = weights_list.cuda()

    return weights_list, weights_raw_list

def cal_weights_via_CAN_ou(X, Z, num_neighbors, links=0):
    weights_list1 = torch.zeros((X.shape[0], X.shape[2], X.shape[2]))
    weights_raw_list1 = torch.zeros((X.shape[0], X.shape[2], X.shape[2]))
    softmax = torch.nn.Softmax(dim=1)
    for i in range(X.shape[0]):
        size = X[i].shape[1]
        # print(X[i].shape)
        distances = distance(X[i,Z[i,:],:], X[i,Z[i,:],:])
        raw_weights = softmax(-distances)
        weights = (raw_weights+raw_weights.t())/2
        weights_list1[i] = weights
        weights_raw_list1[i] = raw_weights
    weights_raw_list1 = weights_raw_list1.cuda()
    weights_list1 = weights_list1.cuda()
    return weights_raw_list1,weights_list1

def cal_weights_via_CAN(X, num_neighbors, links=0):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return:
    """
    weights_list = torch.zeros((X.shape[0],X.shape[2],X.shape[2]))
    weights_raw_list = torch.zeros((X.shape[0], X.shape[2], X.shape[2]))
    for i in range(X.shape[0]):
       size = X[i].shape[1]
       #print(X[i].shape)
       distances = distance(X[i], X[i])
       distances = torch.max(distances, torch.t(distances))
       sorted_distances, _ = distances.sort(dim=1)
       top_k = sorted_distances[:, num_neighbors]
       top_k = torch.t(top_k.repeat(size, 1)) + 10**-10

       sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
       sum_top_k = torch.t(sum_top_k.repeat(size, 1))
       sorted_distances = None
       torch.cuda.empty_cache()
       T = top_k - distances
       distances = None
       torch.cuda.empty_cache()
       weights = torch.div(T, num_neighbors * top_k - sum_top_k)
       T = None
       top_k = None
       sum_top_k = None
       torch.cuda.empty_cache()
       weights = weights.relu().cpu()
       if links is not 0:
           links = torch.Tensor(links).cuda()
           weights += torch.eye(size).cuda()
           weights += links
           weights /= weights.sum(dim=1).reshape([size, 1])
       torch.cuda.empty_cache()
       raw_weights = weights
       weights = (weights + weights.t()) / 2
       weights_list[i] = weights
       weights_raw_list[i] = raw_weights
    weights_raw_list = weights_raw_list.cuda()
    weights_list = weights_list.cuda()

    return weights_list, weights_raw_list


def get_Laplacian_from_weights(weights):
    # W = torch.eye(weights.shape[0]).cuda() + weights
    # degree = torch.sum(W, dim=1).pow(-0.5)
    # return (W * degree).t()*degree
    Laplacian_list = torch.zeros((weights.shape[0],weights.shape[1],weights.shape[1]))
    for i in range(weights.shape[0]):
        degree = torch.sum(weights[i], dim=1).pow(-0.5)
        Laplacian_list[i]=(weights[i] * degree).t()*degree
    return Laplacian_list

def get_Laplacian_from_weights_xin(weights):
    degree = torch.sum(weights, dim=1).pow(-0.5)
    return (weights * degree).t()*degree
def noise(weights, ratio=0.1):
    sampling = torch.rand(weights.shape).cuda() + torch.eye(weights.shape[0]).cuda()
    sampling = (sampling > ratio).type(torch.IntTensor).cuda()
    return weights * sampling


if __name__ == '__main__':
    tX = torch.rand(3, 8)
    print(cal_weights_via_CAN(tX, 3))
