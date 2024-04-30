import sys, os
import time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sconfig import returncfg
from network import SpectralNet
from utils import ineedtosaveresults, get_results, label2img, imgsave
from datatools import get_data, DataLoader
from sklearn.preprocessing import *


cfg = returncfg()
# 去除背景并打乱的数据，去除背景并打乱的标签，打乱位置索引，原始标签
ori_input_data, labels, label_, index = get_data(datasetsname=cfg['dset'], cfg=cfg)
# input_data = ori_input_data
# input_data = MinMaxScaler().fit_transform(ori_input_data)
input_data = StandardScaler().fit_transform(ori_input_data)

cfg['n_clusters'] = len(np.unique(labels))
cfg['use_kmeans'] = True
mresults_idx = 0

output_data = input_data
# 孪生数据集
siamdataset = (output_data, labels), (output_data, labels)
print(siamdataset[0][0].shape)
output_data = []
cfg['data_dim'] = np.shape(siamdataset[0][0])[-1]
data_loader = DataLoader(cfg, siamdataset)
cfg['num'] = siamdataset[0][0].shape[0]
spectralnet = SpectralNet(cfg)
# mmmodel = spectralnet.train(data_loader)
train_model = spectralnet.train_dscrle(data_loader, label_, cfg, index)
# output_data = spectralnet.predict(data_loader)
# mresult, predictions = get_results(output_data, labels, cfg)

# indian
# OA
# Kappa
# pur
# ari
# ami
# fmi
# nmi
# 157
# 0.4888
# 0.4350
# 0.5683
# 0.2985
# 0.5090
# 0.3761
# 0.5113


# epoch
# OA
# Kappa
# pur
# ari
# ami
# fmi
# nmi
# 35
# 0.6129
# 0.5832
# 0.6253
# 0.4361
# 0.6269
# 0.4811
# 0.6279


