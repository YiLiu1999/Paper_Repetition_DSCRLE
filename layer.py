import torch.nn as nn
import torch.nn.functional as f
import torch
import math
import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")


# 实现了正交化矩阵的功能，对输入矩阵进行特征变换，输出正交的特征矩阵
class Orthonorm(nn.Module):
    # 初始化函数，其中    units     为特征维度
    def __init__(self, units):
        super(Orthonorm, self).__init__()
        # 1024
        self.units = units

    # 前向传播函数，其中    inputs    为输入的特征矩阵，epsilon    为浮点数，用于调整矩阵的对角线元素。
    def forward(self, inputs, epsilon=1e-6):
        x = inputs
        # 计算输入矩阵X的转置矩阵X_T和X的矩阵乘积，得到 X_T * X
        x_2 = torch.matmul(torch.transpose(x, 0, 1), x)
        # 计算输入矩阵X的转置矩阵X_T和X的矩阵乘积，得到 X_T * X
        x_2 = x_2 + (torch.eye(x_2.shape[0]) * epsilon).to(device)
        # 通过  Cholesky  分解求得正定矩阵X_T * X的下三角矩阵L，用于进行全部特征值的分解
        L = torch.linalg.cholesky(x_2)
        # 通过计算 L 的逆矩阵，得到正交变换矩阵O，使得 $X ^ T * X = O ^ T * O$，并乘上特征维度的平方根，用于保证新特征矩阵的方差与原始特征矩阵相同。
        ortho_weights = torch.transpose(torch.linalg.inv(L), 0, 1) * torch.sqrt(
            torch.tensor(self.units, dtype=torch.float32))

        return torch.matmul(x, ortho_weights)


def make_layer_list(arch, network_type=None, reg=None, dropout=0):
    '''
    Generates the list of layers specified by arch, to be stacked
    by stack_layers (defined in src/core/layer.py)

    arch:           list of dicts, where each dict contains the arguments
                    to the corresponding layer function in stack_layers

    network_type:   siamese or spectral net. used only to name layers

    reg:            L2 regularization (if any)
    dropout:        dropout (if any)

    returns:        appropriately formatted stack_layers dictionary
    '''
    layers = []
    for i, a in enumerate(arch):
        layer = {'l2_reg': reg}
        layer.update(a)
        if network_type:
            layer['name'] = '{}_{}'.format(network_type, i)
        layers.append(layer)
        if a['type'] != 'Flatten' and dropout != 0:
            dropout_layer = {
                'type': nn.Dropout(p=dropout),
            }
            if network_type:
                dropout_layer['name'] = '{}_dropout_{}'.format(network_type, i)
            layers.append(dropout_layer)
    return layers


def stack_layers(layers):
    slayers = []
    for layer in layers:
        # check for l2_reg argument
        l2_reg = layer.get('l2_reg')
        if l2_reg:
            l2_reg = f.l2_loss()

        # create the layer
        if layer['type'] == 'Dense':
            l = nn.Linear(layer.get('size1'), layer.get('size2'))
            slayers.append(l)
            if layer.get('activation') == 'relu':
                l = nn.ReLU()
            else:
                l = nn.Tanh()
            slayers.append(l)
            if l2_reg:
                l.weight_regularizer = l2_reg
        elif layer['type'] == 'Conv2D':
            l = nn.Conv2d(500, layer['channels'], kernel_size=layer['kernel'], padding=layer.get('padding', 0))
            slayers.append(l)
            l = nn.ReLU()
            slayers.append(l)
            if l2_reg:
                l.weight_regularizer = l2_reg
        elif layer['type'] == 'BatchNormalization':
            l = nn.BatchNorm1d(layer.get('size2'))
            slayers.append(l)
        elif layer['type'] == 'MaxPooling2D':
            l = nn.MaxPool2d(kernel_size=layer.get('pool_size'))
            slayers.append(l)
        elif layer['type'] == 'Dropout':
            l = nn.Dropout(p=layer.get('rate'))
            slayers.append(l)
        elif layer['type'] == 'Flatten':
            l = nn.Flatten()
            slayers.append(l)
        elif layer['type'] == 'Orthonorm':
            l = Orthonorm(layer.get('batchsize')).to(device)
            slayers.append(l)
        else:
            raise ValueError("Invalid layer type '{}'".format(layer['type']))

    return slayers


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # L_sym*Z*W

        support = torch.mm(input, self.weight)
        h = torch.mm(adj, support)
        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, cfg):
        super(GCN, self).__init__()

        self.conv1 = GraphConvolution(144, 128)
        self.nom1 = nn.BatchNorm1d(128)
        self.lR1 = nn.LeakyReLU()

        self.conv2 = GraphConvolution(128, 64)
        self.nom2 = nn.BatchNorm1d(64)
        self.lR2 = nn.LeakyReLU()

        self.conv3 = GraphConvolution(64, cfg['n_clusters'])

    def forward(self, x0, adj):
        h1 = self.lR1(self.nom1(self.conv1(x0, adj)))

        h2 = self.lR2(self.nom2(self.conv2(h1, adj)))

        h3 = self.conv3(h2, adj)

        # h = torch.cat([h1, h2, h3], dim=1)

        return h3
