from torch.optim import Adam, RMSprop
from cost import *
from layer import *
import torch.nn as nn
import os
from matplotlib import pyplot as plt
from DSCRLE.src.utils import get_results, Draw_Classification, Draw_tsne
import time

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
device = torch.device("cuda:{}".format(9) if torch.cuda.is_available() else "cpu")
loss_func = nn.MSELoss()


class MyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layerlists = make_layer_list(cfg['arch'], 'spectral', cfg['spec_reg'])
        self.layerlists += [{'type': 'Dense', 'activation': 'tanh', 'size1': 50, 'size2': cfg['n_clusters']}]
        self.layerlists += [{'type': 'Orthonorm', 'name': 'orthonorm', 'batchsize': cfg['batch_size']}]
        self.slayers = nn.Sequential(*stack_layers(self.layerlists))
        self.data_dim = cfg['data_dim']

        self.GCN = GCN(cfg)
        # cluster center
        self.cluster_layer = nn.Parameter(torch.Tensor(cfg['n_clusters'], cfg['n_clusters']))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, inputs, adj):
        input = inputs
        for slayer in self.slayers:
            inputs = slayer(inputs)
        h = self.GCN(input, adj)
        return inputs, h

    def calloss(self, X, y_pred, cfg):
        # mscale = get_scale( X.reshape( (cfg['batch_size'], -1) ), cfg['batch_size'] )
        # AffinityMatrix = full_affinity(torch.reshape(X, [cfg['batch_size'], -1]), scale = mscale)
        if (cfg['affinity'] == 'anchor'):
            AffinityMatrix = anchor_affinity(torch.reshape(X, [cfg['batch_size'], -1]), n_nbrs=cfg['n_nbrs'])
            DYY = squared_distance(y_pred)
            loss = torch.sum(AffinityMatrix * DYY) / torch.tensor(cfg['batch_size'], dtype=torch.float32)

        elif (cfg['affinity'] == 'full'):
            AffinityMatrix = full_affinity(torch.reshape(X, [cfg['batch_size'], -1]))
            DYY = squared_distance(y_pred)
            loss = torch.sum(AffinityMatrix * DYY) / torch.tensor(cfg['batch_size'], dtype=torch.float32)

        elif (cfg['affinity'] == 'knn'):
            AffinityMatrix = knn_affinity(torch.reshape(X, [cfg['batch_size'], -1]), n_nbrs=cfg['n_nbrs'])
            DYY = squared_distance(y_pred)
            loss = torch.sum(AffinityMatrix * DYY) / torch.tensor(cfg['batch_size'], dtype=torch.float32)

        elif (cfg['affinity'] == 'sec'):
            self.gammad = 0.5
            self.miu = 1e-5
            self.a = torch.tensor(cfg['batch_size'], dtype=torch.float32)
            # (224, 16)
            self.Wxbf = torch.zeros([self.data_dim, cfg['n_clusters']], requires_grad=True, dtype=torch.float32).to(
                device)
            AffinityMatrix, self.Wxbf = sec(X, y_pred, cfg['n_nbrs'], self.data_dim, self.gammad, self.Wxbf)
            DYY = squared_distance(y_pred)
            sc_loss = torch.sum(AffinityMatrix * DYY) / self.a
            rlc_loss = self.gammad * torch.sum(torch.norm(torch.matmul(X, self.Wxbf) - y_pred, dim=1)) / self.a \
                       + (1.0 - self.gammad) * torch.sum(torch.norm(self.Wxbf, dim=0)) / self.a

            loss = sc_loss + self.miu * rlc_loss

        elif (cfg['affinity'] == 'kec'):
            self.gammad = 0.5
            self.miu = 1e-5
            self.Wxbf = torch.zeros([self.data_dim, cfg['n_clusters']], requires_grad=True, dtype=torch.float32)
            [AffinityMatrix, self.Wxbf] = kec(X, y_pred, cfg['n_nbrs'], self.data_dim, self.gammad, self.Wxbf)
            DYY = squared_distance(y_pred)
            sc_loss = torch.sum(AffinityMatrix * DYY) / torch.tensor(cfg['batch_size'], dtype=torch.float32)
            secloss = torch.sum(torch.norm(torch.matmul(X, self.Wxbf) - y_pred, dim=1)) / torch.tensor(
                cfg['batch_size'], dtype=torch.float32)
            ct_loss = torch.sum(torch.norm(self.Wxbf, dim=0)) / torch.tensor(cfg['batch_size'], dtype=torch.float32)
            loss = sc_loss + self.miu * self.gammad * secloss + self.miu * (1.0 - self.gammad) * ct_loss

        elif (cfg['affinity'] == 'fec'):
            self.gammad = 0.5
            self.miu = 1e-5
            self.Wxbf = torch.zeros([self.data_dim, cfg['n_clusters']], requires_grad=True, dtype=torch.float32)
            [AffinityMatrix, self.Wxbf] = fec(X, y_pred, cfg['n_nbrs'], self.data_dim, self.gammad, self.Wxbf)
            DYY = squared_distance(y_pred)
            sc_loss = torch.sum(AffinityMatrix * DYY) / torch.tensor(cfg['batch_size'], dtype=torch.float32)
            secloss = torch.sum(torch.norm(torch.matmul(X, self.Wxbf) - y_pred, dim=1)) / torch.tensor(
                cfg['batch_size'], dtype=torch.float32)
            ct_loss = torch.sum(torch.norm(self.Wxbf, dim=0)) / torch.tensor(cfg['batch_size'], dtype=torch.float32)
            loss = sc_loss + self.miu * self.gammad * secloss + self.miu * (1.0 - self.gammad) * ct_loss

        return loss


class SpectralNet():
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = MyModel(self.cfg).to(device)

    def train(self, data_loader):
        print(self.model)
        optimizer = RMSprop(self.model.parameters(), lr=self.cfg['spec_lr'])
        num_batches = int(data_loader.num_train_data // self.cfg['batch_size'] * self.cfg['spec_ne'])
        print(num_batches)

        for batch_index in range(num_batches):
            # (1024, 224)
            X, _ = data_loader.get_batch(self.cfg['batch_size'])
            X = X.to(device)
            AffinityMatrix = anchor_affinity(torch.reshape(X, [self.cfg['batch_size'], -1]), n_nbrs=self.cfg['n_nbrs'])
            y_, h = self.model(X, AffinityMatrix)
            # III = torch.type( torch.matmul( tf.transpose(y_pred), y_pred), dtype=tf.uint8 )
            y = y_ + h
            loss = self.model.calloss(X, y, self.cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("batch %d: loss %f" % (batch_index, loss))

        return self.model

    def train_dscrle(self, data_loader, gt, cfg, index):
        print(self.model)
        optimizer = RMSprop(self.model.parameters(), lr=self.cfg['spec_lr'])
        num_batches = (data_loader.num_test_data + self.cfg['batch_size'] - 1) // self.cfg['batch_size']
        train_acc_save = []
        best_label = torch.zeros(cfg['num'])
        best_feature = torch.zeros(cfg['num'], 16)
        best_acc = 0
        for epoch in range(0, self.cfg['spec_ne']):
            batches = [(i * self.cfg['batch_size'], min(data_loader.num_train_data, (i + 1) * self.cfg['batch_size']))
                       for i in range(num_batches)]
            for i, (batch_start, batch_end) in enumerate(batches):
                X = data_loader.test_data[batch_start:batch_end, :]
                X = X.to(device)
                AffinityMatrix = anchor_affinity(torch.reshape(X, [X.shape[0], -1]), n_nbrs=self.cfg['n_nbrs'])
                y_pred, h = self.model(X, AffinityMatrix)
                y = y_pred + h
                loss = self.model.calloss(X, y, self.cfg)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print("epoch %d: batch %d: loss %f" % (epoch, i, loss))
                if (i == 0):
                    y_preds = np.array(y_pred.cpu().detach().numpy())
                else:
                    y_preds = np.concatenate((np.array(y_preds), np.array(y_pred.cpu().detach().numpy())), axis=0)

            acc, y, mapping, purity, kappa, nmi, ari, ami, fmi = get_results(y_preds, data_loader.train_label, self.cfg)
            if acc > best_acc:
                best_epoch = epoch
                best_label = y
                best_acc = acc
                best_kappa = kappa
                best_pur = purity
                best_nmi = nmi
                best_fmi = fmi
                best_ari = ari
                best_ami = ami
                best_feature = y_preds
            train_acc_save.append(acc)
        print('%10s %10s %10s %10s %10s %10s %10s %10s' % ('epoch', 'OA', 'Kappa', 'pur', 'ari', 'ami', 'fmi', 'nmi'))
        print('%10s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f' % (
            best_epoch,
            best_acc,
            best_kappa,
            best_pur,
            best_ari,
            best_ami,
            best_fmi,
            best_nmi))
        Draw_Classification(best_label[index]+1, gt, cfg['dset'], best_acc)
        # Draw_tsne(best_feature, data_loader.train_label, best_acc, cfg['dset'], title=None)
        return self.model

    def predict(self, data_loader):
        num_batches = (data_loader.num_test_data + self.cfg['batch_size'] - 1) // self.cfg['batch_size']
        batches = [(i * self.cfg['batch_size'], min(data_loader.num_test_data, (i + 1) * self.cfg['batch_size'])) for i
                   in range(num_batches)]
        for i, (batch_start, batch_end) in enumerate(batches):
            X = data_loader.test_data[batch_start:batch_end, :]
            X = X.to(device)
            AffinityMatrix = anchor_affinity(torch.reshape(X, [X.shape[0], -1]), n_nbrs=self.cfg['n_nbrs'])
            y_pred, _ = self.model(X, AffinityMatrix)
            if (i == 0):
                y_preds = np.array(y_pred.cpu().detach().numpy())
            else:
                y_preds = np.concatenate((np.array(y_preds), np.array(y_pred.cpu().detach().numpy())), axis=0)
            # III = torch.type( torch.matmul( tf.transpose(y_pred), y_pred), dtype=tf.uint8 )
        return y_preds
