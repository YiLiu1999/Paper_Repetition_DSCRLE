import matplotlib.pyplot as plt
from scipy import signal
from DSCRLE.src.HSIdata import *
import torch


def get_data(datasetsname=None, cfg=0):
    if (datasetsname == 'har'):
        data, label = load_har()
    elif (datasetsname == 'indiapines'):
        data, label = load_IndianPines()
    elif (datasetsname == 'Botswana'):  # 145
        data, label = load_Botswana()
    elif (datasetsname == 'salinas'):  # 102
        data, label = load_salinas()
    elif (datasetsname == 'paviaU'):  # 103
        data, label = load_paviauni()
    elif (datasetsname == 'houstonu'):  # 103
        data, label = load_houstonu()
    elif (datasetsname == 'houstonu18'):  # 103
        data, label = load_houstonu18()
    elif (datasetsname == 'hanchuan'):
        data, label = load_hanchuan()
    elif (datasetsname == 'honghu'):
        data, label = load_honghu()
    elif (datasetsname == 'longkou'):
        data, label = load_longkou()
    elif (datasetsname == 'salinas'):
        data, label = load_saline()
    else:
        data, label = None
    # plt.title("pred" + "_label")
    # plt.imshow(label)
    # plt.show()
    data = np.array(data, dtype='float32')
    # 计算图像梯度
    if (cfg['spatial'] > 1):
        maskss = cfg['spatial'] ** 2
        in2 = np.array(np.ones(shape=(cfg['spatial'], cfg['spatial'])) / maskss)
        mdata = np.empty_like(data)
        for i in range(np.shape(data)[-1]):
            mdata[:, :, i] = signal.convolve2d(data[:, :, i], in2, 'same')
        # (1476, 256, 145)
        data = mdata

    if (len(np.shape(data)) > 2):
        traindatashape = np.shape(data)
        # （377856， 145）
        x_train = data.reshape([-1, traindatashape[-1]])
        # （377856）
        y_train = label.flatten()
        # 去除背景
        # (3248, 145)
        x_train = x_train[np.nonzero(y_train)]
        y_train = y_train[np.nonzero(y_train)]
    else:
        x_train = data
        y_train = label
    c = y_train
    index = np.random.choice(x_train.shape[0], x_train.shape[0], replace=False)
    x_train = x_train[index, :]
    y_train = y_train[index]
    sorted_indexes = sorted(range(len(index)), key=lambda i: index[i])
    return x_train, y_train, label, sorted_indexes


class DataLoader():
    def __init__(self, cfg=None, datasets=None):
        (self.train_data, self.train_label), (self.test_data, self.test_label) = datasets
        self.train_data = self.train_data.astype(np.float32)
        self.test_data = torch.from_numpy(self.test_data.astype(np.float32))
        self.train_label = self.train_label.astype(np.int32)
        self.test_label = torch.from_numpy(self.test_label.astype(np.int32))
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, self.num_train_data, batch_size)
        return torch.from_numpy(self.train_data[index, :]), torch.from_numpy(self.train_label[index])
