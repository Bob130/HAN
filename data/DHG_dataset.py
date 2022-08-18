#!/usr/bin/env python  
# -*- coding: utf-8 -*-
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import random
import math
import pickle
import json

import numpy as np

import torch
from torch.utils.data import Dataset


class DHGDataset(Dataset):
    """writing posture dataset."""

    def __init__(self, json_file, class_num, input_frames, resolution, model, gap=3, is_train=False, test_subject_id=1,
                 transform=None):
        """
        Args:
            data_path (string): skeleton file path.
            label_file (string): label file path
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.json_file = json_file
        self.class_num = class_num
        self.input_frames = input_frames
        self.resolution = resolution
        self.model = model
        self.gap = gap  # 每个骨架的间隔
        self.is_train = is_train
        self.test_subject_id = test_subject_id
        self.transform = transform

        self.data = None
        self.num_joint = 22
        self.rotate_ratio = 0.0
        self.max_rotate_angle = 10
        self.noise_ratio = 0.5
        self.max_noise = 0.01
        self.shift_ratio = 0.0
        self.max_shift = 0.05
        self.finger_tip_idx = [5, 9, 13, 17, 21]
        # http://www-rech.telecom-lille.fr/shrec2017-hand/
        # 1.Wrist, 2.Palm, 3.thumb_base, 4.thumb_first_joint, 5.thumb_second_joint, 6.thumb_tip, 7.index_base,
        # 8.index_first_joint, 9.index_second_joint, 10.index_tip, 11.middle_base, 12.middle_first_joint,
        # 13.middle_second_joint, 14.middle_tip, 15.ring_base, 16.ring_first_joint, 17.ring_second_joint, 18.ring_tip,
        # 19.pinky_base, 20.pinky_first_joint, 21.pinky_second_joint, 22.pinky_tip.

        self.sigma = 1.0

        # load label
        with open(self.json_file, 'r') as f:
            all_data = json.load(f)
            self.data = []
            if self.is_train:
                for item in all_data:
                    if item['id_subject'] != self.test_subject_id:
                        self.data.append(item)
            else:
                for item in all_data:
                    if item['id_subject'] == self.test_subject_id:
                        self.data.append(item)

        self.global_max_delta_xyz = np.array([0.27649599, 0.26539327, 0.202666])
        self.edge_len = self.global_max_delta_xyz

        # compute max delta xyz
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # self.global_max_delta_xyz = np.array([0.0, 0.0, 0.0])
        # # load data
        # for sample_idx, sample in enumerate(all_data):
        #     data = np.array(sample['skeletons'])
        #
        #     for i in range(np.size(data, 0)):
        #         skeleton = data[i, :]
        #         skeleton_xyz = skeleton.reshape(self.num_joint, 3)
        #         max_xyz = skeleton_xyz.max(axis=0)
        #         min_xyz = skeleton_xyz.min(axis=0)
        #         delta_xyz = max_xyz - min_xyz
        #         if delta_xyz[0] > 0.23:
        #             print('delta_x > 2\n')
        #         self.global_max_delta_xyz = np.maximum(self.global_max_delta_xyz, delta_xyz)
        # --------------------------------------------------------------------------------------------------------------
        print(self.global_max_delta_xyz)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(idx)
        data, data_frames = self.get_skeleton_sequence(idx)     # data_frames <= np.size(data, 0)

        if self.class_num == 14:
            label = self.data[idx]['labels_14'] - 1
        elif self.class_num == 28:
            label = self.data[idx]['labels_28'] - 1

        if self.is_train:
            if random.random() < self.rotate_ratio:
                angle_y = self.max_rotate_angle * random.uniform(-1.0, 1.0) * math.pi / 180
                for i in range(data_frames):
                    skeleton = data[i, :, :]

                    skeleton = torch.from_numpy(skeleton)  # tensor and ndarray share the same memory
                    skeleton = self.rotate(skeleton, angle_y)
                    data[i, :, :] = skeleton.numpy()
            if random.random() < self.noise_ratio:
                noise = np.random.uniform(low=-1.0, high=1.0, size=data.shape[1:3]) * self.max_noise
                for i in range(data_frames):
                    skeleton = data[i, :, :]
                    # noise = np.random.uniform(low=-1.0, high=1.0, size=data.shape[1:3]) * self.max_noise
                    skeleton = skeleton + noise
                    data[i, :, :] = skeleton
            if random.random() < self.shift_ratio:
                for i in range(data_frames):
                    skeleton = data[i, :, :]
                    shift = np.random.uniform(low=-1.0, high=1.0, size=3) * self.max_shift
                    skeleton = skeleton + shift
                    data[i, :, :] = skeleton

        # sequence_boundingbox_max_xyz = np.max(data, 1)
        # sequence_boundingbox_min_xyz = np.min(data, 1)
        # sequence_boundingbox_max = sequence_boundingbox_max_xyz - sequence_boundingbox_min_xyz
        # sequence_boundingbox_max = np.max(sequence_boundingbox_max, 0)
        # sequence_cubeboundingbox_edge = np.max(sequence_boundingbox_max, 0)

        centres = np.mean(data, 1)
        # data_frames = np.size(centres, 0)
        move_dir = np.zeros_like(centres)
        move_dir[1:data_frames, :] = centres[1:data_frames, :] - centres[0:data_frames-1, :]
        move_dir[move_dir > 0] = 1.0
        move_dir[move_dir < 0] = -1.0

        centres[0:data_frames, :] = centres[0:data_frames, :] - centres[0, :]

        finger_tip = data[:, self.finger_tip_idx, :].copy()
        finger_tip[0:data_frames, :, :] = finger_tip[0:data_frames, :, :] - finger_tip[0, :, :]

        thumb_tip_vector = data[:, self.finger_tip_idx, :].copy()
        thumb_tip_vector = thumb_tip_vector.transpose(1, 0, 2)
        thumb_tip_vector = thumb_tip_vector - thumb_tip_vector[0, :, :]
        thumb_tip_vector = thumb_tip_vector.transpose(1, 0, 2)[:, 1::, :]

        image = np.concatenate((np.expand_dims(centres, 1), finger_tip), 1)
        image = image.transpose((2, 0, 1))

        centres = torch.from_numpy(centres).to(torch.float32)
        centres = centres.view(-1)
        move_dir = torch.from_numpy(move_dir).to(torch.float32)
        move_dir = move_dir.view(-1)
        finger_tip = torch.from_numpy(finger_tip).to(torch.float32)
        finger_tip = finger_tip.view(-1)
        thumb_tip_vector = torch.from_numpy(thumb_tip_vector).to(torch.float32)
        thumb_tip_vector = thumb_tip_vector.view(-1)
        image = torch.from_numpy(image).to(torch.float32)

        # centres = torch.cat([centres, finger_tip], 0)
        centres = image
        finger_tip = thumb_tip_vector

        if self.model == 'c3d':
            input_3d = torch.zeros(1, self.resolution+self.gap*(self.input_frames-1), self.resolution, self.resolution)
        elif self.model == 'lstm':
            input_3d = torch.zeros(self.input_frames, 1, self.resolution, self.resolution, self.resolution)

        for i in range(data_frames):
            skeleton = data[i, :, :].copy()
            skeleton = torch.from_numpy(skeleton)
            skeleton = self.isotropic_normalize_3d(skeleton)
            # skeleton = self.local_normalize_3d_coordinate(skeleton)
            # skeleton = self.sequence_max_cubeboundingbox_normalize(skeleton, sequence_cubeboundingbox_edge)
            skeleton = self.sparse3d(skeleton)

            if self.model == 'c3d':
                skeleton[:, 0].add_(i * self.gap)
                index_i = torch.zeros([skeleton.shape[0], skeleton.shape[1]+1], dtype=torch.long)
                index_i[:, 1:4] = skeleton
                index_i.t_()
                input_3d[index_i.tolist()] = 1

                # gaussian3d_i = self.gaussian3d(skeleton)
                # index_from = self.gap*i
                # index_to = self.gap*i+self.resolution
                # input_3d[0, index_from:index_to, :, :] = torch.max(input_3d[0, index_from:index_to, :, :], gaussian3d_i)
            elif self.model == 'lstm':
                for j in range(0, skeleton.size()[0]):
                    input_3d[i, 0, skeleton[j, 0], skeleton[j, 1], skeleton[j, 2]] = 1

                # gaussian3d_i = self.gaussian3d(skeleton)
                # input_3d[i, 0, :, :, :] = gaussian3d_i

        # input_3d.unsqueeze_(0)  # ch=1, d, h, w

        label = torch.tensor(label)

        data_idx = {'joint_cloud': input_3d, 'centres': centres, 'finger_tip': finger_tip, 'label': label, 'index': idx}

        if self.transform:
            data_idx = self.transform(data_idx)

        return data_idx

    def get_skeleton_sequence(self, idx):
        sample_idx = idx
        sample = self.data[sample_idx]

        # load the idx th data
        skeleton_sequence = sample['skeletons']
        beginning_frame = sample['beginning_frame']
        end_frame = sample['end_frame']
        frames = end_frame - beginning_frame + 1
        # use effective frames
        skeleton_sequence = skeleton_sequence[beginning_frame:end_frame+1]

        data = np.zeros((self.input_frames, self.num_joint, 3))

        if frames > self.input_frames:
            data_frames = self.input_frames
            step = (frames-1) / self.input_frames

            for i in range(self.input_frames):
                frame_from = math.ceil(i * step)
                frame_to = math.floor((i+1) * step)
                # ratio = random.random() if self.is_train else 0.5
                ratio = 0.5
                frame_idx = round(frame_from + ratio * (frame_to - frame_from))
                skeleton_i = np.array(skeleton_sequence[frame_idx]).reshape(self.num_joint, 3)
                data[i, :, :] = skeleton_i
        elif frames <= self.input_frames:
            # pad 0
            # data_frames = frames
            # for i in range(frames):
            #     skeleton_i = np.array(skeleton_sequence[i]).reshape(self.num_joint, 3)
            #     data[i, :, :] = skeleton_i

            # repeat
            data_frames = self.input_frames
            step = (frames-1) / self.input_frames
            for i in range(self.input_frames):
                frame_idx = math.floor((i+1) * step)
                skeleton_i = np.array(skeleton_sequence[frame_idx]).reshape(self.num_joint, 3)
                data[i, :, :] = skeleton_i

        return data, data_frames

    @staticmethod
    def local_normalize(data):
        min_xyz = data.min(dim=0)[0]
        max_xyz = data.max(dim=0)[0]
        ((data[:, 0].sub_(min_xyz[0])).div_(max_xyz[0] - min_xyz[0]))  # data normalization, x~[0,1]
        ((data[:, 1].sub_(min_xyz[1])).div_(max_xyz[1] - min_xyz[1]))  # data normalization, y~[0,1]
        ((data[:, 2].sub_(min_xyz[2])).div_(max_xyz[2] - min_xyz[2]))  # data normalization, z~[0,1]
        return data

    @staticmethod
    def sequence_max_cubeboundingbox_normalize(data, bounding_box_edge):
        min_xyz = data.min(dim=0)[0]
        max_xyz = data.max(dim=0)[0]
        ((data[:, 0].sub_((min_xyz[0] + max_xyz[0]) / 2.0)).div_(bounding_box_edge)).add_(0.5)  # x normalization
        ((data[:, 1].sub_((min_xyz[1] + max_xyz[1]) / 2.0)).div_(bounding_box_edge)).add_(0.5)  # y normalization
        ((data[:, 2].sub_((min_xyz[2] + max_xyz[2]) / 2.0)).div_(bounding_box_edge)).add_(0.5)  # z normalization
        data[data.gt(1.0)] = 1.0
        data[data.lt(0.0)] = 0.0
        return data

    def isotropic_normalize_3d(self, data):
        min_xyz = data.min(dim=0)[0]
        max_xyz = data.max(dim=0)[0]
        ((data[:, 0].sub_((min_xyz[0] + max_xyz[0]) / 2.0)).div_(self.edge_len[0])).add_(0.5)  # x normalization
        ((data[:, 1].sub_((min_xyz[1] + max_xyz[1]) / 2.0)).div_(self.edge_len[1])).add_(0.5)  # y normalization
        ((data[:, 2].sub_((min_xyz[2] + max_xyz[2]) / 2.0)).div_(self.edge_len[2])).add_(0.5)  # z normalization
        data[data.gt(1.0)] = 1.0
        data[data.lt(0.0)] = 0.0
        return data

    @staticmethod
    def vector3d(self, data):
        data = data.view(-1)
        return data

    def sparse3d(self, data):
        # 3D input
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        resolution_3d = self.resolution  # 3D输入的空间分辨率
        data.mul_(resolution_3d - 1).round_()
        data = data.to(torch.long)
        # input_3d = torch.zeros(resolution_3d, resolution_3d, resolution_3d)
        # for i in range(0, data.size()[0]):
        #     input_3d[data[i, 0], data[i, 1], data[i, 2]] = 1
        # --------------------------------------------------------------------------------------------------------------

        # input_3d.unsqueeze_(0)  # ch=1, d, h, w
        return data

    def gaussian3d(self, data):
        # 3D input
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        resolution_3d = self.resolution     # 3D输入的空间分辨率
        sigma = self.sigma
        # data.mul_(resolution_3d-1).round_()
        input_3d = torch.zeros(resolution_3d, resolution_3d, resolution_3d)
        for i in range(data.size()[0]):
            point_x, point_y, point_z = data[i][0], data[i][1], data[i][2]
            x = torch.arange(0.0, resolution_3d)
            y = torch.arange(0.0, resolution_3d)
            z = torch.arange(0.0, resolution_3d)
            grid_x, grid_y, grid_z = torch.meshgrid([x, y, z])

            gaussian_3d = torch.exp( (-1/(2*sigma**2)) * ( (grid_x-point_x).pow(2) + (grid_y-point_y).pow(2) + (grid_z-point_z).pow(2) ) )
            input_3d = torch.max(input_3d, gaussian_3d)
            # input_3d.add_(gaussian_3d)
        # --------------------------------------------------------------------------------------------------------------

        # input_3d.unsqueeze_(0)      # ch=1, d, h, w
        return input_3d

    @staticmethod
    def rotate(data, angle_y):
        """Rotate randomly the image in a sample.
        Args:
            data (tensor): skeleton data
            angle_y (float): The rotate angle along y axis.
        """
        center = torch.mean(data, 0)
        center_pose = data - center

        rotate_y = torch.tensor([[math.cos(angle_y), 0, math.sin(angle_y)], [0, 1, 0], [-math.sin(angle_y), 0, math.cos(angle_y)]],
                                dtype=torch.float64)
        rotate_center_pose = torch.transpose(center_pose, 0, 1)
        rotate_center_pose = torch.mm(rotate_y, rotate_center_pose)
        rotate_pose = rotate_center_pose.transpose(0, 1) + center

        # rotate_min_xyz = rotate_pose.min(dim=0)[0]
        # rotate_max_xyz = rotate_pose.max(dim=0)[0]
        # if rotate_min_xyz.lt(0.0).sum().item() > 0 or rotate_max_xyz.gt(1.0).sum().item() > 0:
        #     return data
        # else:
        #     return rotate_pose

        return rotate_pose

    @staticmethod
    def show_skeleton(skeleton):
        x = skeleton[:, 0]
        y = skeleton[:, 1]
        z = skeleton[:, 2]

        # plt.ion()
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot(x, y, z, 'bo')
        ax.view_init(elev=90, azim=90)  # 改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('r')
        plt.savefig('debug.png')
        # plt.show()

        # plt.ioff()

    @staticmethod
    def view_invariant_angle(data):
        if np.size(data, 1) > 1:
            joint_move = []
            for b in range(np.size(data, 1)):
                joint_move_b = np.zeros(np.size(data, 2), 3)
                for i in range(1, 10):
                    joint_move_b += np.abs(data[i, b, :, 0:3] - data[i-1, b, :, 0:3])
                joint_move.append(joint_move_b)
            main_actor = joint_move.index(max(joint_move))

        for i in range(np.size(data, 0)):
            skeleton = data[i, main_actor, :, :]
            left_shoulder = skeleton[4, :]
            right_shoulder = skeleton[8, :]
            if left_shoulder[3] > 1.5 and right_shoulder[3] > 1.5:
                pass

            skeleton = skeleton[skeleton[:, 3] > 1.5, 0:3]
            if skeleton.size == 0:
                continue

