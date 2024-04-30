import cv2
from sklearn import cluster, metrics
from scipy.optimize import linear_sum_assignment as linear_assignment
import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import adjusted_mutual_info_score as ami_score
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import fowlkes_mallows_score as fmi_score
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


def mkmeans(X, nclusters):
    # X = StandardScaler().fit_transform(X)
    # X = MinMaxScaler().fit_transform(X)
    k_means = cluster.MiniBatchKMeans(n_clusters=nclusters)
    k_means.fit(X)
    y_pred = k_means.predict(X)
    return y_pred


def msc(X, params):
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
    spectral.fit(X)
    y_pred = spectral.labels_.astype(int)
    return y_pred


def cluster_purity(y_true, y_pred):
    """
    Calculate clustering purity
    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        purity, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    w = np.transpose(metrics.confusion_matrix(y_true, y_pred))
    label_mapping = w.argmax(axis=1)
    y_pred_voted = y_pred.copy()
    for i in range(y_pred.size):
        y_pred_voted[i] = label_mapping[y_pred[i]]
    return metrics.accuracy_score(y_pred_voted, y_true)


def cluster_kappa(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    pe_rows = np.sum(cm, axis=0)
    pe_cols = np.sum(cm, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)

    w = np.transpose(cm)
    row_ind, col_ind = linear_assignment(w.max() - w)
    sum_right = w[row_ind, col_ind].sum()
    po = sum_right / float(sum_total)

    return (po - pe) / (1 - pe)


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    w = np.transpose(metrics.confusion_matrix(y_true, y_pred))
    row_ind, col_ind = linear_assignment(w.max() - w)
    return np.sum([w[row_ind[i], col_ind[i]] for i in range(0, len(row_ind))]) * 1.0 / y_pred.size


def get_results(y_preds, y_true, cfg):
    if cfg['use_kmeans']:
        y_preds_kmeans = mkmeans(y_preds, cfg['n_clusters'])
        # y_preds_kmeans = y_preds.argmax(1)
    else:
        y_preds_kmeans = y_preds
    acc, y, mapping, purity, kappa, nmi, ari, ami, fmi = eva(y_true, y_preds_kmeans)
    return acc, y, mapping, purity, kappa, nmi, ari, ami, fmi


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def Draw_Classification(pred, label, name, acc, scale: float = 4.0, dpi: int = 400):
    colors = ["black", "yellow", "lightgreen", "indigo", "orange", "pink", "peru", "crimson", "aqua", "dodgerblue",
              "slategrey", "b", "red", "darkcyan", "grey", "olive", "green", "gold"]
    indices = np.where(label != 0)
    label[indices] = pred
    # 创建一个空白的RGB图像，形状与label相同
    rgb_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)

    # 将颜色值赋给每个类别对应的像素
    for i, color in enumerate(colors):
        rgb = np.array(mcolors.to_rgb(color)) * 255
        rgb_label[label == i] = rgb.astype(np.uint8)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.imshow(rgb_label)  # 显示彩色图像
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)

    foo_fig = plt.gcf()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(
        'DSCRLE/results/{}'
        .format(name) + '/pred_{:.5f}'.format(acc) + '.png', format='png',
        transparent=True, dpi=dpi, pad_inches=0)
    plt.show()


def Draw_tsne(X, y, acc, dataname, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    color = ["yellow", "lightgreen", "indigo", "orange", "pink", "peru", "crimson", "aqua", "dodgerblue",
             "slategrey", "b", "red", "darkcyan", "grey", "olive", "green", "gold"]
    for i in range(X.shape[0]):
        if y[i] == 0:
            s0 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[0])
        if y[i] == 1:
            s1 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[1])
        if y[i] == 2:
            s2 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[2])
        if y[i] == 3:
            s3 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[3])
        if y[i] == 4:
            s4 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[4])
        if y[i] == 5:
            s5 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[5])
        if y[i] == 6:
            s6 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[6])
        if y[i] == 7:
            s7 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[7])
        if y[i] == 8:
            s8 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[8])
        if y[i] == 9:
            s9 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[9])
        if y[i] == 10:
            s10 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[10])
        if y[i] == 11:
            s11 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[11])
        if y[i] == 12:
            s12 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[12])
        if y[i] == 13:
            s13 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[13])
        if y[i] == 14:
            s14 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[14])
        if y[i] == 15:
            s15 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[15])
    plt.xlabel('t-SNE:dimension 1')
    plt.ylabel('t-SNE:dimension 2')
    if title is not None:
        plt.title(title)
    if dataname == 'indian':
        plt.legend((s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15),
                   ("Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees",
                    "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
                    "Soybean-clean",
                    "Wheat",
                    "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"), loc='best')
    plt.savefig(
        '/DSCRLE/results/{}'
        .format(dataname) + '/tsne_{:.5f}'.format(acc) + '.png', format='png',
        transparent=True, pad_inches=0)
    plt.show()


def cluster_ac(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id
        num_classes: total number of classes in your dataset

    Returns: acc and f1-score
    """
    y_true = torch.tensor(y_true) - torch.min(torch.tensor(y_true))
    l1 = list(set(y_true.tolist()))
    num_class1 = len(l1)
    y_pred = torch.tensor(y_pred)
    l2 = list(set(y_pred.tolist()))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred.tolist()))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return

    cost = torch.zeros((num_class1, numclass2), dtype=torch.int32)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # 使用 SciPy 的 linear_sum_assignment 执行 Munkres 算法
    cost_np = cost.numpy()
    row_ind, col_ind = linear_sum_assignment(-cost_np)
    new_predict = torch.zeros(len(y_pred))

    mapping = {}  # 用于建立真实标签到预测标签的映射关系
    for i, c in enumerate(l1):
        c2 = l2[col_ind[i]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
        mapping[c2] = c
    y_true = y_true.cpu()
    acc = metrics.accuracy_score(y_true, new_predict)

    matrix = confusion_matrix(y_true, new_predict)
    # 选择每个簇中最大的数值
    max_cluster_values = np.max(matrix, axis=0)
    # 计算purity
    purity = np.sum(max_cluster_values) / np.sum(matrix)
    ka = kappa(y_true.cpu().numpy(), new_predict.cpu().numpy())
    nmi = nmi_score(y_true.cpu().numpy(), new_predict.cpu().numpy())
    ami = ami_score(y_true.cpu().numpy(), new_predict.cpu().numpy())
    ari = ari_score(y_true.cpu().numpy(), new_predict.cpu().numpy())
    fmi = fmi_score(y_true.cpu().numpy(), new_predict.cpu().numpy())
    return acc, new_predict, mapping, purity, ka, nmi, ari, ami, fmi


def eva(y_true, y_pred):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    acc, y, mapping, purity, kappa, nmi, ari, ami, fmi = cluster_ac(y_true, y_pred)
    print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ami {:.4f}'.format(ami),
          ', ari {:.4f}'.format(ari),
          ', fmi {:.4f}'.format(fmi), ', kappa {:.4f}'.format(kappa), ', purity {:.4f}'.format(purity))
    return acc, y, mapping, purity, kappa, nmi, ari, ami, fmi


def make_batches(size, batch_size):
    '''
    generates a list of (start_idx, end_idx) tuples for batching data
    of the given size and batch_size

    size:       size of the data to create batches for
    batch_size: batch size

    returns:    list of tuples of indices for data
    '''
    num_batches = (size + batch_size - 1) // batch_size  # round up
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(num_batches)]


def ineedtosaveresults(name, data=None, filepath='result.txt', moshi='a'):
    f = open(filepath, moshi)
    if (data == None):
        f.write(name)
        f.write('\n')
    elif (type(data) == 'dict'):
        for key, value in data.items():
            f.write(key + ':' + str(value))
            f.write('\n')
    elif (type(data) == 'list'):
        f.write(name + ':' + str(data))
        f.write('\n')
    else:
        f.write(name + ':' + str(data))
        f.write('\n')
    f.close()


def label2img(inpredict, inlabel):
    label_flatten = inlabel.flatten()
    if (len(inpredict) == len(label_flatten)):
        predictions = label_flatten
    else:
        predictions = np.zeros(np.shape(label_flatten))
        predictions[np.nonzero(label_flatten)] = inpredict

    predictions = predictions.reshape(np.shape(inlabel))

    if (len(np.shape(inlabel)) > 1):
        label_arr = cv2.cvtColor(np.uint8(inlabel / inlabel.max() * 255.0), cv2.COLOR_GRAY2RGB)
        label_arr = cv2.applyColorMap(label_arr, cv2.COLORMAP_JET)

        predictions_arr = cv2.cvtColor(np.uint8(predictions / predictions.max() * 255.0), cv2.COLOR_GRAY2RGB)
        predictions_arr = cv2.applyColorMap(predictions_arr, cv2.COLORMAP_JET)
    else:
        label_arr = inlabel
        predictions_arr = predictions

    return label_arr, predictions_arr


def imgsave(label_arr, predictions_arr, save_path, metheds):
    cv2.imwrite(save_path + '/' + metheds + '_label_arr.png', label_arr)
    cv2.imwrite(save_path + '/' + metheds + '_predictions_arr.png', predictions_arr)
    return True


def resultshow(label_arr, predictions_arr):
    cv2.imshow('label_arr', label_arr)
    cv2.imshow('predictions_arr', predictions_arr)
    return True

#
# cfg = returncfg()
# data, label, label_ = get_data('indiapines')
# cfg['n_clusters'] = len( np.unique(label) )
#
# minmaxscaler = MinMaxScaler()
# data = minmaxscaler.fit_transform(data)
# # data = StandardScaler().fit_transform(data)
# metric_list = ['acc','purity','rand_score','adjusted_rand_score','normalized_mutual_info_score','adjusted_mutual_info_score','v_measure_score']
# results, y_preds_kmeans = get_results(data, label, cfg)
# # label_arr, predictions_arr = label2img(results, label)
# # resultshow(label_arr, predictions_arr)
# print(metric_list)
# print(results)
