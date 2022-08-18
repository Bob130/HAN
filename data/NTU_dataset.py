import os
import sys
import numpy as np
import pickle
import torch
import random
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, Dataset

edge = ((0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5),
        (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0),
        (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
        (19, 18), (21, 22), (22, 7), (23, 24), (24, 11))


class Skeleton(Dataset):
    def __init__(self, data_path, label_path, window_size, final_size,
                 mode='train', random_choose=False, center_choose=False):
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.random_choose = random_choose
        self.center_choose = center_choose
        self.window_size = window_size
        self.final_size = final_size
        self.edge = None
        self.load_data()

    def load_data(self):
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = int(self.label[index])
        sample_name = self.sample_name[index]
        data_numpy = np.array(data_numpy)  # nctv

        data_numpy = data_numpy[:, data_numpy.sum(0).sum(-1).sum(-1) != 0]  # CTVM

        # # data transform
        # if self.decouple_spatial:
        #     data_numpy = decouple_spatial(data_numpy, edges=self.edge)
        # if self.num_skip_frame is not None:
        #     velocity = decouple_temporal(data_numpy, self.num_skip_frame)
        #     C, T, V, M = velocity.shape
        #     data_numpy = np.concatenate((velocity, np.zeros((C, 1, V, M))), 1)

        # data_numpy = pad_recurrent_fix(data_numpy, self.window_size)  # if short: pad recurrent
        # data_numpy = uniform_sample_np(data_numpy, self.window_size)  # if long: resize
        if self.random_choose:
            data_numpy = random_sample_np(data_numpy, self.window_size)
            # data_numpy = random_choose_simple(data_numpy, self.final_size)
        else:
            data_numpy = uniform_sample_np(data_numpy, self.window_size)
        if self.center_choose:
            # data_numpy = uniform_sample_np(data_numpy, self.final_size)
            data_numpy = random_choose_simple(data_numpy, self.final_size, center=True)
        else:
            data_numpy = random_choose_simple(data_numpy, self.final_size)

        # # random change subject1 and subject2
        # if random.random() < 0.5:
        #     tmp = np.zeros_like(data_numpy)
        #     tmp[:, :, :, 0] = data_numpy[:, :, :, 1]
        #     tmp[:, :, :, 1] = data_numpy[:, :, :, 0]
        #     data_numpy = tmp

        data_idx = {'input': torch.from_numpy(data_numpy).float(), 'label': torch.tensor(label), 'index': index}
        return data_idx
        # if self.mode == 'train':
        #     return data_numpy.astype(np.float32), label
        # else:
        #     return data_numpy.astype(np.float32), label, sample_name

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def vis(data, edge, is_3d=True, pause=0.01, view=0.25, title=''):
    import os

    os.environ['DISPLAY'] = 'localhost:10.0'
    import matplotlib.pyplot as plt
    import matplotlib

    # matplotlib.use('Qt5Agg')
    C, T, V, M = data.shape

    plt.ion()
    fig = plt.figure()
    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    ax.set_title(title)
    p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
    import sys
    from os import path
    sys.path.append(
        path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    pose = []
    for m in range(M):
        a = []
        for i in range(len(edge)):
            if is_3d:
                a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
            else:
                a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
        pose.append(a)
    ax.axis([-view, view, -view, view])
    if is_3d:
        ax.set_zlim3d(-view, view)
    for t in range(T):
        for m in range(M):
            for i, (v1, v2) in enumerate(edge):
                x1 = data[:2, t, v1, m]
                x2 = data[:2, t, v2, m]
                if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                    pose[m][i].set_xdata(data[0, t, [v1, v2], m])
                    pose[m][i].set_ydata(data[1, t, [v1, v2], m])
                    if is_3d:
                        pose[m][i].set_3d_properties(data[2, t, [v1, v2], m])
        fig.canvas.draw()
        plt.pause(pause)
    plt.close()
    plt.ioff()


def uniform_sample_np(data_numpy, size):
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    interval = T / size
    uniform_list = [int(i * interval) for i in range(size)]
    return data_numpy[:, uniform_list]


def random_sample_np(data_numpy, size):
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    interval = int(np.ceil(size / T))
    random_list = sorted(random.sample(list(range(T))*interval, size))
    return data_numpy[:, random_list]


def random_choose_simple(data_numpy, size, center=False):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if size < 0:
        assert 'resize shape is not right'
    if T == size:
        return data_numpy
    elif T < size:
        return data_numpy
    else:
        if center:
            begin = (T - size) // 2
        else:
            begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


class NTU_SKE(Skeleton):
    def __init__(self, data_path, label_path, window_size, final_size, mode='train',
                 random_choose=False, center_choose=False):
        super().__init__(data_path, label_path, window_size, final_size, mode,
                         random_choose, center_choose)
        self.edge = edge

    def load_data(self):
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        self.data = np.load(self.data_path, mmap_mode='r')[:, :3]  # NCTVM


def test(data_path, label_path, vid=None, edge=None, is_3d=False, mode='train'):
    dataset = NTU_SKE(data_path, label_path, window_size=48, final_size=32, mode=mode,
                      random_choose=True, center_choose=False)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    labels = open('../prepare/ntu_120/label.txt', 'r').readlines()
    for i, (data, label) in enumerate(loader):
        if i%1000==0:
            vis(data[0].numpy(), edge=edge, view=1, pause=0.01, title=labels[label.item()].rstrip())

    sample_name = loader.dataset.sample_name
    sample_id = [name.split('.')[0] for name in sample_name]
    index = sample_id.index(vid)
    if mode != 'train':
        data, label, index = loader.dataset[index]
    else:
        data, label = loader.dataset[index]
    # skeleton
    vis(data, edge=edge, view=1, pause=0.1)


if __name__ == '__main__':
    data_path = "/your/path/to/ntu/xsub/val_data_joint.npy"
    label_path = "/your/path/to/ntu/xsub/val_label.pkl"
    test(data_path, label_path, vid='S004C001P003R001A032', edge=edge, is_3d=True, mode='train')
    # data_path = "/your/path/to/ntu/xsub/val_data_joint.npy"
    # label_path = "/your/path/to/ntu/xsub/val_label.pkl"
    # test(data_path, label_path, vid='S004C001P003R001A032', edge=edge, is_3d=True, mode='train')
