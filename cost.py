import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
device = torch.device("cuda:{}".format(9) if torch.cuda.is_available() else "cpu")


def get_scale(x, batch_size, n_nbrs=10):
    '''
    Calculates the scale* based on the median distance of the kth
    neighbors of each point of x*, a m-sized sample of x, where
    k = n_nbrs and m = batch_size

    x:          data for which to compute scale
    batch_size: m in the aforementioned calculation. it is
                also the batch size of spectral net
    n_nbrs:     k in the aforementeiond calculation.

    returns:    the scale*

    *note:      the scale is the variance term of the gaussian
                affinity matrix used by spectral net
    '''
    n = len(x)

    # sample a random batch of size batch_size
    sample = x[np.random.randint(n, size=batch_size), :]
    # flatten it
    sample = sample.reshape((batch_size, np.prod(sample.shape[1:])))

    # compute distances of the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_nbrs).fit(sample)
    distances, _ = nbrs.kneighbors(sample)

    # return the median distance
    return np.median(distances[:, n_nbrs - 1])


def squared_distance(X, Y=None):
    '''
    Calculates the pairwise distance between points in X and Y

    X:          n x d matrix
    Y:          m x d matrix
    W:          affinity -- if provided, we normalize the distance

    returns:    n x m matrix of all pairwise squared Euclidean distances
    '''
    if Y is None:
        Y = X
    X1 = torch.reshape(X, (1, X.shape[0], X.shape[-1]))
    Y1 = torch.reshape(Y, (Y.shape[0], 1, Y.shape[-1]))
    DXY = torch.sum(torch.square(X1 - Y1), dim=-1)

    return DXY


def full_affinity(X, Y=None, scale=1.0):
    DXX = squared_distance(X)
    scale = torch.tensor(scale, dtype=torch.float32)
    sigma = scale
    sigma_squared = torch.pow(sigma, 2)
    sigma_squared = torch.unsqueeze(sigma_squared, -1).to(device)
    Dx_scaled = DXX / (2 * sigma_squared)
    S = torch.exp(-Dx_scaled)
    return S


def knn_affinity(X, n_nbrs, scale=None, scale_nbr=None, local_scale=True, verbose=False):
    '''
    Calculates the symmetrized Gaussian affinity matrix with k1 nonzero
    affinities for each point, scaled by
    1) a provided scale,
    2) the median distance of the k2th neighbor of each point in X, or
    3) a covariance matrix S where S_ii is the distance of the k2th
    neighbor of each point i, and S_ij = 0 for all i != j
    Here, k1 = n_nbrs, k2=scale_nbr

    X:              input dataset of size n
    n_nbrs:         k1
    scale:          provided scale
    scale_nbr:      k2, used if scale not provided
    local_scale:    if True, then we use the aforementioned option 3),
                    else we use option 2)
    verbose:        extra printouts

    returns:        n x n affinity matrix
    '''
    import torch

    if isinstance(n_nbrs, np.float):
        n_nbrs = int(n_nbrs)
    elif isinstance(n_nbrs, torch.Tensor) and n_nbrs.dtype != torch.int32:
        n_nbrs = n_nbrs.int()
    # get squared distance
    Dx = squared_distance(X)
    # calculate the top k neighbors of minus the distance (so the k closest neighbors)
    nn_val, nn_ind = torch.topk(-Dx, k=n_nbrs, dim=-1)

    vals = nn_val
    # apply scale
    if scale is None:
        # if scale not provided, use local scale
        if scale_nbr is None:
            scale_nbr = 0
        else:
            print("getAffinity scale_nbr, n_nbrs:", scale_nbr, n_nbrs)
            assert scale_nbr > 0 and scale_nbr <= n_nbrs
        if local_scale:
            scale = -nn_val[:, scale_nbr - 1]
            scale = torch.reshape(scale, [-1, 1])
            scale = scale.repeat(1, n_nbrs)
            scale = torch.reshape(scale, [-1, 1])
            vals = torch.reshape(vals, [-1, 1])
            if verbose:
                vals = torch.cat((torch.reshape(nn_val, [-1, 1]), torch.reshape(scale, [-1, 1])), axis=1)
                print("vals, scale shape", vals.shape)
            vals = vals / (2 * scale)
            vals = torch.reshape(vals, [-1, n_nbrs])
        else:
            def get_median(scales, m):
                with torch.no_grad():
                    scales, _ = torch.topk(scales, k=m, dim=-1)
                scale = scales[:, m - 1]
                return scale, scales

            scales = -nn_val[:, scale_nbr - 1]
            const = X.shape[0] // 2
            scale, scales = get_median(scales, const)
            vals = vals / (2 * scale)
    else:
        # otherwise, use provided value for global scale
        vals = vals / (2 * scale ** 2)

    # get the affinity
    affVals = torch.exp(vals)
    # flatten this into a single vector of values to shove in a spare matrix
    affVals = torch.reshape(affVals, [-1])
    # get the matrix of indexes corresponding to each rank with 1 in the first column and k in the kth column
    nnInd = nn_ind
    # get the J index for the sparse matrix
    jj = torch.reshape(nnInd, [-1, 1])
    # the i index is just sequential to the j matrix
    ii = torch.arange(nnInd.shape[0]).reshape(-1, 1)
    ii = ii.repeat((1, nnInd.shape[1]))
    ii = torch.reshape(ii, [-1, 1])
    # concatenate the indices to build the sparse matrix
    indices = torch.cat((ii, jj), dim=1)
    # assemble the sparse Weight matrix
    S = torch.sparse_coo_tensor(indices=indices.T.to(dtype=torch.int64), values=affVals,
                                size=torch.tensor(Dx.shape, dtype=torch.int64))
    # fix the ordering of the indices
    S = S.coalesce()
    # convert to dense tensor
    S = S.to_dense()
    # symmetrize
    S = (S + torch.transpose(S, 0, 1)) / 2.0;

    return S


def anchor_affinity(X, n_nbrs):
    # 计算输入特征矩阵X中每个数据点之间的距离的平方，得到距离矩阵Dx
    Dx = squared_distance(X)
    # 将距离矩阵Dx中的距离值取负，得到相似度矩阵D
    D = -Dx
    # 对相似度矩阵D中每个数据点的相似度进行排序，选出每个数据点的最近邻点。其中，Dsort(1024, 4)为排序后的相似度矩阵，Didx(1024, 4)为排序后相似度矩阵对应的数据点索引。
    [Dsort, Didx] = torch.topk(D, n_nbrs + 1, dim=-1)
    # 根据 Dsort 和 Didx 计算锚点图的矩阵 S。
    # (1024, 1024)
    S = get_S_achor(Dsort, Didx, n_nbrs).to(device)
    return S


def sec(X, F, n_nbrs, data_dim, gammad, Wxbf):
    S = anchor_affinity(X, n_nbrs)
    # gammad=0.5
    wf1 = torch.inverse(torch.matmul(X.T, X) + (((1 - gammad) / gammad) * torch.eye(data_dim)).to(device))
    wf2 = torch.matmul(X.T, F)
    W = torch.matmul(wf1, wf2)
    Wmoment = 0.9 * Wxbf.to(device) + 0.1 * W
    return S, Wmoment


def kec(X, F, n_nbrs, data_dim, gammad, Wxbf):
    S = knn_affinity(X, n_nbrs)

    wf1 = torch.inverse(torch.matmul(X.T, X) + (((1 - gammad) / gammad) * torch.eye(data_dim)).to(device))
    wf2 = torch.matmul(X.T, F)
    W = torch.matmul(wf1, wf2)
    Wmoment = 0.9 * Wxbf.to(device) + 0.1 * W
    return S, Wmoment


def fec(X, F, n_nbrs, data_dim, gammad, Wxbf):
    S = full_affinity(X, n_nbrs)

    wf1 = torch.inverse(torch.matmul(X.T, X) + (((1 - gammad) / gammad) * torch.eye(data_dim)).to(device))
    wf2 = torch.matmul(X.T, F)
    W = torch.matmul(wf1, wf2)
    Wmoment = 0.9 * Wxbf.to(device) + 0.1 * W
    return S, Wmoment


# Dsort 与像素最接近的neighr个像素的距离；Didx相应的索引
def get_S_achor(Dsort, Didx, neibor=11):
    # 将相似度矩阵 D 中的每个元素取负，得到距离矩阵.
    Dsort = -Dsort
    # torch.Size([1024, 4])
    Xshape = Dsort.size()
    # 从最近邻点的索引矩阵 Didx 取出每个数据点的前 neibor 个最近邻点，得到一个 N×(neibor - 1)的矩阵 colidx，其中 neibor是一个超参数，表示每个数据点选择的最近邻数目.
    colidx = Didx[:, 1:neibor]
    # 将 colidx 在张量的最后一个维度上增加一个维度，以便与 rowidx 进行拼接。
    colidx = colidx.unsqueeze(axis=-1)
    # 首先使用 torch.arange 创建一个从0到N - 1的序列，然后将它在第二个和第三个维度上都添加一个额外的维度，最后将该序列转换为一个大小为N×(neibor-1)×1的张量rowidx，以便与colidx进行拼接。
    rowidx = torch.arange(0, Xshape[0]).unsqueeze(axis=-1).unsqueeze(axis=-1).to(device)
    # 将rowidx在第二个维度上复制neibor - 1次，得到一个N×(neibor - 1)×1的张量rowidx，以便与colidx进行拼接。
    rowidx = rowidx.repeat(1, colidx.size()[-2], 1)
    # 将rowidx 和 colidx沿着最后一个维度进行拼接，并将结果张量从一个大小为N×(neibor - 1)×2的张量变形为一个大小为N×(neibor - 1) * 2的张量idx，其中每一行表示一个边的起始和结束点的索引。
    idx = torch.reshape(torch.cat([rowidx, colidx], dim=-1), [-1, 2])
    # 从距离矩阵中取出每个数据点的前neibor个最近邻点的相似度，得到一个N×(neibor - 1)的矩阵valuek。
    # k近邻个点
    valuek = -Dsort[:, 1:neibor]
    # 从距离矩阵中取出每个数据点的第neibor个最近邻点的相似度，得到一个大小为N的向量valuekk。
    # 锚点
    valuekk = -Dsort[:, neibor]
    # 将valuekk在第二个维度上重复neibor - 1次，得到一个N×(neibor - 1)的矩阵valuekk，以便与valuek进行计算。
    valuekk = torch.tile(valuekk.unsqueeze(axis=-1), [1, valuek.size()[-1]])
    # 计算valuek沿着倒数第一个维度上的和，N×1的向量valuesum，用于计算Svalue。
    valuesum = torch.sum(valuek, axis=-1, keepdims=True)
    # 将valuesum在第二个维度上重复neibor - 1次，得到一个N×(neibor - 1)的矩阵valuesum，以便与valuek和valuekk进行计算。
    valuesum = torch.tile(valuesum, [1, valuek.size()[-1]])
    # 将valuek变形为一个大小为(N * (neibor - 1))的向量。
    valuek = torch.reshape(valuek, [-1])
    # 将valuekk变形为一个大小为(N * (neibor - 1))的向量。
    valuekk = torch.reshape(valuekk, [-1])
    # 将valuesum变形为一个大小为(N * (neibor - 1))的向量。
    valuesum = torch.reshape(valuesum, [-1])
    # 根据公式计算每个边的S值。
    Svalue = (valuekk - valuek) / (valuekk * (neibor - 1) - valuesum + torch.tensor(1e-7))
    # 创建一个大小为N×N的全零张量S。
    S = torch.zeros(Xshape[0], Xshape[0]).to(device)
    # 将计算得到的S值赋值给S中对应边的位置。
    S[idx[:, 0], idx[:, 1]] = Svalue
    # 同样将计算得到的S值赋值S中对称位置的边。
    S[idx[:, 1], idx[:, 0]] = Svalue
    # 将S变为对称矩阵。
    S = (S + torch.transpose(S, 0, 1)) / 2.0
    return S


def euclidean_distance(vects):
    x, y = vects
    sum_square = torch.sum(torch.square(x - y), dim=1, keepdims=True)
    return torch.sqrt(torch.max(sum_square, torch.tensor([1e-7])))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def compute_accuracy(y_true, y_pred):  # numpy上的操作
    '''Compute classification accuracy with a fixed threshold on distances.'''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):  # Tensor上的操作
    '''Compute classification accuracy with a fixed threshold on distances.'''
    return torch.mean(torch.eq(y_true, torch.cast(y_pred < 0.5, y_true.dtype)))
